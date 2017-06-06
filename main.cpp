#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <string>
#include <assert.h>
#include <random>
#include <math.h>

#include <CL/cl2.hpp>

#define LEARN_RATE 0.05

#define LAYER_ST 0
#define LAYER_FC 1
#define LAYER_CV 2
#define LAYER_RE 3
#define LAYER_PL 4

namespace nn
{
	template<typename T>
	void vpack(std::vector<T>& v,std::ofstream& out)
	{
		out.write((char*)&(v[0]),v.size()*sizeof(T));
	}
	template<typename T>
	void writeout(std::ofstream& out,T&&val)
	{
		out.write((char*)&val,sizeof(T));
	}
	template<typename T>
	void readin(std::ifstream& in,T* val,int mul=1)
	{
		in.read((char*)val,sizeof(T)*mul);
	}
	cl_int3 make_int3(int a,int b,int c)
	{
		cl_int3 tmp;
		tmp.s[0]=a;
		tmp.s[1]=b;
		tmp.s[2]=c;
		return tmp;
	}
	cl_int4 make_int4(int a,int b,int c,int d)
	{
		cl_int4 tmp;
		tmp.s[0]=a;
		tmp.s[1]=b;
		tmp.s[2]=c;
		tmp.s[3]=d;
		return tmp;
	}

	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<double> dist(0.0001,0.001);
				
	cl::Program program;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Kernel fc_feedforward_kernel;
	cl::Kernel fc_internalize_error_kernel;
	cl::Kernel fc_backpropagate_kernel;
	cl::Kernel fc_adapt_kernel;
	cl::Kernel cv_feedforward_kernel;
	cl::Kernel cv_adapt_kernel;
	cl::Kernel cv_backpropagate_kernel;
	cl::Kernel relu_feedforward_kernel;
	cl::Kernel relu_backpropagate_kernel;
	cl::Kernel pool_feedforward_kernel;
	cl::Kernel pool_backpropagate_kernel;
	cl::Kernel IEC_kernel;
	void init()
	{
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		std::vector<cl::Device> devices;
		platforms[0].getDevices(CL_DEVICE_TYPE_GPU,&devices);
		cl::Device gpu=devices[0];
		std::cout<<"Using GPU \""<<gpu.getInfo<CL_DEVICE_NAME>()<<"\""<<std::endl;
		std::vector<cl::Device> context_devices;
		context_devices.push_back(gpu);
		context=cl::Context(context_devices);
		std::ifstream sourceFile("main.cl");
		std::string sourceCode(std::istreambuf_iterator<char>(sourceFile),(std::istreambuf_iterator<char>()));
		cl::Program::Sources src;
		src.push_back(sourceCode);
		program=cl::Program(context,src);
		program.build(context_devices);
		std::cout<<"Build log: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(gpu)<<std::endl;
		queue=cl::CommandQueue(context, gpu);
		fc_feedforward_kernel=cl::Kernel(program,"fc_feedforward_kernel");
		fc_internalize_error_kernel=cl::Kernel(program,"fc_internalize_error_kernel");
		fc_backpropagate_kernel=cl::Kernel(program,"fc_backpropagate_kernel");
		fc_adapt_kernel=cl::Kernel(program,"fc_adapt_kernel");
		cv_feedforward_kernel=cl::Kernel(program,"cv_feedforward_kernel");
		cv_adapt_kernel=cl::Kernel(program,"cv_adapt_kernel");
		cv_backpropagate_kernel=cl::Kernel(program,"cv_backpropagate_kernel");
		relu_feedforward_kernel=cl::Kernel(program,"relu_feedforward_kernel");
		relu_backpropagate_kernel=cl::Kernel(program,"relu_backpropagate_kernel");
		pool_feedforward_kernel=cl::Kernel(program,"pool_feedforward_kernel");
		pool_backpropagate_kernel=cl::Kernel(program,"pool_backpropagate_kernel");
		IEC_kernel=cl::Kernel(program,"initial_error_calc_kernel");
	}
	class layer
	{
	// protected:
	public:
		std::vector<int> dimensions;
		int total_size;
		cl::Buffer activation_buffer;
		cl::Buffer error_buffer;
		layer *previous,*next=NULL;
		virtual ~layer(){ }
		virtual void feedforward()=0;
		virtual void learn()=0;
		virtual void pack(std::ofstream& out)=0;
	};
	class relu_layer: public layer
	{
	public:
		cl::Kernel& ff_k;
		cl::Kernel& bp_k;
		relu_layer(layer* p): ff_k(relu_feedforward_kernel),bp_k(relu_backpropagate_kernel)
		{
			previous=p;
			p->next=this;
			dimensions=p->dimensions;
			total_size=p->total_size;
			activation_buffer=cl::Buffer(context,CL_MEM_READ_WRITE,total_size*sizeof(double));
			error_buffer=cl::Buffer(context,CL_MEM_READ_WRITE,total_size*sizeof(double));
		}
		void feedforward()
		{
			ff_k.setArg(0,previous->activation_buffer);
			ff_k.setArg(1,activation_buffer);
			ff_k.setArg(2,make_int3(dimensions[0],dimensions[1],dimensions[2]));
			queue.enqueueNDRangeKernel(ff_k,cl::NullRange,cl::NDRange(dimensions[0],dimensions[1],dimensions[2]));			
			if(next)
				next->feedforward();
		}
		void learn()
		{
			bp_k.setArg(0,previous->error_buffer);
			bp_k.setArg(1,error_buffer);
			bp_k.setArg(2,previous->activation_buffer);
			bp_k.setArg(3,make_int3(dimensions[0],dimensions[1],dimensions[2]));
			queue.enqueueNDRangeKernel(bp_k,cl::NullRange,cl::NDRange(dimensions[0],dimensions[1],dimensions[2]));
			previous->learn();
		}
		void pack(std::ofstream& out)
		{
			writeout(out, LAYER_RE);
		}
	};
	class pool_layer: public layer
	{
	public:
		cl::Kernel& ff_k;
		cl::Kernel& bp_k;
		int dim=2;
		pool_layer(layer* p): ff_k(pool_feedforward_kernel),bp_k(pool_backpropagate_kernel)
		{
			previous=p;
			p->next=this;
			dimensions=p->dimensions;
			dimensions[0]=(dimensions[0]-1)/dim+1;
			dimensions[1]=(dimensions[1]-1)/dim+1;
			total_size=dimensions[0]*dimensions[1]*dimensions[2];
			activation_buffer=cl::Buffer(context,CL_MEM_READ_WRITE,total_size*sizeof(double));
			error_buffer=cl::Buffer(context,CL_MEM_READ_WRITE,total_size*sizeof(double));			
		}
		void feedforward()
		{
			ff_k.setArg(0,previous->activation_buffer);
			ff_k.setArg(1,activation_buffer);
			ff_k.setArg(2,make_int3(dimensions[0],dimensions[1],dimensions[2]));
			ff_k.setArg(3,dim);
			queue.enqueueNDRangeKernel(ff_k,cl::NullRange,cl::NDRange(dimensions[0],dimensions[1],dimensions[2]));
			if(next)
				next->feedforward();
		}
		void learn()
		{
			bp_k.setArg(0,previous->error_buffer);
			bp_k.setArg(1,error_buffer);
			bp_k.setArg(2,previous->activation_buffer);
			bp_k.setArg(3,activation_buffer);
			bp_k.setArg(4,make_int3(dimensions[0],dimensions[1],dimensions[2]));
			bp_k.setArg(5,dim);
			queue.enqueueNDRangeKernel(bp_k,cl::NullRange,cl::NDRange(dimensions[0],dimensions[1],dimensions[2]));
			previous->learn();
		}
		void pack(std::ofstream& out)
		{
			writeout(out,LAYER_PL);
		}
	};
	class conv_layer: public layer
	{
	public:
		//Notice: no biases in conv layer. I'm not sure if it's good or not, but most probably it is
		std::vector<double> filters;
		cl::Buffer filters_buffer;
		int filter_dim;
		int filter_num;
		cl::Kernel& ff_k;
		cl::Kernel& bp_k;
		cl::Kernel& ad_k;
		conv_layer(int f_d,int f_n,layer* p,std::ifstream* fs=NULL): ff_k(cv_feedforward_kernel),bp_k(cv_backpropagate_kernel),ad_k(cv_adapt_kernel)
		{
			filter_dim=f_d;
			filter_num=f_n;
			previous=p;
			p->next=this;
			dimensions=p->dimensions;
			assert(dimensions.size()==3);
			dimensions[2]=filter_num;
			total_size=dimensions[0]*dimensions[1]*dimensions[2];
			int tmp_size=(filter_dim*filter_dim*filter_num)*filter_num;//(filter_size)*num_of_filters
			filters=std::vector<double>(tmp_size);
			activation_buffer=cl::Buffer(context,CL_MEM_READ_WRITE,total_size*sizeof(double));
			error_buffer=cl::Buffer(context,CL_MEM_READ_WRITE,total_size*sizeof(double));
			filters_buffer=cl::Buffer(context,CL_MEM_READ_WRITE,tmp_size*sizeof(double));
			if(fs)
			{
				std::ifstream& ifs=*fs;
				readin(ifs,&(filters[0]),tmp_size);
			}
			else
			{
				for(int i=0;i<tmp_size;++i)
					filters[i]=dist(e2);
			}
			queue.enqueueWriteBuffer(filters_buffer,CL_FALSE,0,filters.size()*sizeof(double),(double*)&(filters[0]));
		}
		void feedforward()
		{
			ff_k.setArg(0,previous->activation_buffer);
			ff_k.setArg(1,activation_buffer);
			ff_k.setArg(2,filters_buffer);
			ff_k.setArg(3,filter_dim);
			ff_k.setArg(4,make_int3(dimensions[0],dimensions[1],dimensions[2]));
			ff_k.setArg(5,make_int3(previous->dimensions[0],previous->dimensions[1],previous->dimensions[2]));
			queue.enqueueNDRangeKernel(ff_k,cl::NullRange,cl::NDRange(dimensions[0],dimensions[1],dimensions[2]));
			if(next)
				next->feedforward();
		}
		void learn()
		{
			bp_k.setArg(0,previous->error_buffer);
			bp_k.setArg(1,error_buffer);
			bp_k.setArg(2,filters_buffer);
			bp_k.setArg(3,filter_dim);
			bp_k.setArg(4,make_int3(previous->dimensions[0],previous->dimensions[1],previous->dimensions[2]));
			bp_k.setArg(5,make_int3(dimensions[0],dimensions[1],dimensions[2]));
			queue.enqueueNDRangeKernel(bp_k,cl::NullRange,cl::NDRange(previous->dimensions[0],previous->dimensions[1],previous->dimensions[2]));
			previous->learn();
			ad_k.setArg(0,filters_buffer);
			ad_k.setArg(1,error_buffer);
			ad_k.setArg(2,previous->activation_buffer);
			ad_k.setArg(3,make_int3(dimensions[0],dimensions[1],dimensions[2]));
			ad_k.setArg(4,make_int3(previous->dimensions[0],previous->dimensions[1],previous->dimensions[2]));
			ad_k.setArg(5,filter_dim);
			ad_k.setArg(6,LEARN_RATE);
			queue.enqueueNDRangeKernel(ad_k,cl::NullRange,cl::NDRange(filter_dim,filter_dim,previous->dimensions[2]));
		}
		void pack(std::ofstream& out)
		{
			queue.enqueueReadBuffer(filters_buffer,CL_TRUE,0,filters.size()*sizeof(double),(double*)&(filters[0]));
			writeout(out,LAYER_CV);
			writeout(out,filter_dim);
			writeout(out,filter_num);
			vpack(filters,out);
		}
	};
	class fc_layer: public layer
	{
	// protected:
	public:
		std::vector<double> weights;
		cl::Buffer weights_buffer;
		std::vector<double> biases;
		cl::Buffer biases_buffer;
		cl::Buffer w_inp_buffer;
		cl::Buffer IEC_buffer;
		cl::Kernel& ff_k;
		cl::Kernel& bp_k;
		cl::Kernel& ie_k;
		cl::Kernel& ad_k;
		fc_layer(std::vector<int> n_d_s,layer* p,std::ifstream* fs=NULL): ff_k(fc_feedforward_kernel),bp_k(fc_backpropagate_kernel),ie_k(fc_internalize_error_kernel),ad_k(fc_adapt_kernel)
		{
			dimensions=n_d_s;
			assert(dimensions.size()==3);
			total_size=1;
			for(int i: dimensions)
				total_size*=i;
			int pts=p->total_size;
			activation_buffer=cl::Buffer(context,CL_MEM_READ_WRITE,total_size*sizeof(double));
			error_buffer=cl::Buffer(context,CL_MEM_READ_WRITE,total_size*sizeof(double));
			biases_buffer=cl::Buffer(context,CL_MEM_READ_WRITE,total_size*sizeof(double));
			w_inp_buffer=cl::Buffer(context,CL_MEM_READ_WRITE,total_size*sizeof(double));
			weights_buffer=cl::Buffer(context,CL_MEM_READ_WRITE,pts*total_size*sizeof(double));
			previous=p;
			p->next=this;
			weights=std::vector<double>(total_size*pts);
			biases=std::vector<double>(total_size);
			if(fs)
			{
				std::ifstream& ifs=*fs;
				readin(ifs,&(biases[0]),biases.size());
				readin(ifs,&(weights[0]),weights.size());
			}
			else
			{
				for(unsigned int i=0;i<biases.size();++i)
					biases[i]=dist(e2);
				for(unsigned int i=0;i<weights.size();++i)
					weights[i]=dist(e2);
			}
			//now load weights and biases
			//Non-blocking because we won't need it modified right away
			queue.enqueueWriteBuffer(weights_buffer,CL_FALSE,0,weights.size()*sizeof(double),(double*)&(weights[0]));
			queue.enqueueWriteBuffer(biases_buffer,CL_FALSE,0,biases.size()*sizeof(double),(double*)&(biases[0]));
			IEC_buffer=cl::Buffer(context,CL_MEM_READ_WRITE,total_size*sizeof(double));
		}
		void pack(std::ofstream& out)
		{
			writeout(out,LAYER_FC);
			queue.enqueueReadBuffer(biases_buffer,CL_FALSE,0,biases.size()*sizeof(double),(double*)&(biases[0]));
			queue.enqueueReadBuffer(weights_buffer,CL_TRUE,0,weights.size()*sizeof(double),(double*)&(weights[0]));
			vpack(dimensions,out);
			vpack(biases,out);
			vpack(weights,out);
		}
		void prepare(double* desired_out)
		{
			IEC_kernel.setArg(0,activation_buffer);
			IEC_kernel.setArg(1,IEC_buffer);
			IEC_kernel.setArg(2,error_buffer);
			IEC_kernel.setArg(3,w_inp_buffer);
			queue.enqueueWriteBuffer(IEC_buffer,CL_FALSE,0,total_size*sizeof(double),desired_out);
			queue.enqueueNDRangeKernel(IEC_kernel,cl::NullRange,cl::NDRange(total_size));
			// queue.enqueueReadBuffer(error_buffer,CL_TRUE,0,total_size*sizeof(double),testarr.data());
			// std::cout<<"External error derivative:\n";
			// for(int i=0;i<total_size;++i)
			// 	std::cout<<testarr[i]<<" ";
			// std::cout<<"\n";
		}
		void unload(double* where)
		{
			queue.enqueueReadBuffer(activation_buffer,CL_TRUE,0,total_size*sizeof(double),where);
		}
		void feedforward()
		{
			ff_k.setArg(0,previous->activation_buffer);
			ff_k.setArg(1,activation_buffer);
			ff_k.setArg(2,w_inp_buffer);
			ff_k.setArg(3,biases_buffer);
			ff_k.setArg(4,weights_buffer);
			ff_k.setArg(5,make_int3(dimensions[0],dimensions[1],dimensions[2]));
			ff_k.setArg(6,make_int3(previous->dimensions[0],previous->dimensions[1],previous->dimensions[2]));
			queue.enqueueNDRangeKernel(ff_k,cl::NullRange,cl::NDRange(dimensions[0],dimensions[1],dimensions[2]));
			if(next)
				next->feedforward();
		}
		void learn()
		{
			ie_k.setArg(0,error_buffer);
			ie_k.setArg(1,w_inp_buffer);
			ie_k.setArg(2,make_int3(dimensions[0],dimensions[1],dimensions[2]));
/*			if(total_size==1)
			{
				double test;
				queue.enqueueReadBuffer(error_buffer,CL_TRUE,0,sizeof(double),&test);
				std::cout<<"External error derivative:\n";
				std::cout<<test<<"\n";
				queue.enqueueNDRangeKernel(ie_k,cl::NullRange,cl::NDRange(dimensions[0],dimensions[1],dimensions[2]));
				queue.enqueueReadBuffer(error_buffer,CL_TRUE,0,sizeof(double),&test);
				std::cout<<"Internal error derivative:\n";
				std::cout<<test<<"\n\n";
			}			*/
			//else
			queue.enqueueNDRangeKernel(ie_k,cl::NullRange,cl::NDRange(dimensions[0],dimensions[1],dimensions[2]));
			bp_k.setArg(0,previous->error_buffer);
			bp_k.setArg(1,error_buffer);
			bp_k.setArg(2,weights_buffer);
			bp_k.setArg(3,make_int3(previous->dimensions[0],previous->dimensions[1],previous->dimensions[2]));
			bp_k.setArg(4,make_int3(dimensions[0],dimensions[1],dimensions[2]));
			queue.enqueueNDRangeKernel(bp_k,cl::NullRange,cl::NDRange(previous->dimensions[0],previous->dimensions[1],previous->dimensions[2]));
			previous->learn();
			ad_k.setArg(0,error_buffer);
			ad_k.setArg(1,previous->activation_buffer);
			ad_k.setArg(2,weights_buffer);
			ad_k.setArg(3,biases_buffer);
			ad_k.setArg(4,make_int3(dimensions[0],dimensions[1],dimensions[2]));
			ad_k.setArg(5,make_int3(previous->dimensions[0],previous->dimensions[1],previous->dimensions[2]));
			ad_k.setArg(6,LEARN_RATE);
			queue.enqueueNDRangeKernel(ad_k,cl::NullRange,cl::NDRange(dimensions[0],dimensions[1],dimensions[2]));
		}
	};
	class frontlayer: public layer
	{
	public:
		std::vector<double> activation;
		frontlayer(std::vector<int> n_d_s)
		{
			dimensions=n_d_s;
			assert(dimensions.size()==3);
			total_size=1;
			for(int i: dimensions)
				total_size*=i;
			activation=std::vector<double>(total_size);
			activation_buffer=cl::Buffer(context,CL_MEM_READ_ONLY,total_size*sizeof(double));
			error_buffer=cl::Buffer(context,CL_MEM_READ_WRITE,total_size*sizeof(double));
		}
		void pack(std::ofstream& out)
		{
			writeout(out,LAYER_ST);
			vpack(dimensions,out);
		}
		void feedforward()
		{
			//blocking so we can call it in a loop
			queue.enqueueWriteBuffer(activation_buffer,CL_TRUE,0,total_size*sizeof(double),(double*)&(activation[0]));
			if(next)
				next->feedforward(); //loads in and passes execution further
		}
		void setup(std::vector<double>&inp)
		{
			assert(activation.size()==inp.size());
			activation=inp;
		}
		void learn()
		{/*really nothing to learn here*/}
	};
	class network
	{
	public:
		std::list<layer*> layers;
		frontlayer* input;
		layer* output;
		void launch(std::vector<double> inp)
		{
			input->setup(inp);
			input->feedforward();
		}
		void train(double* desired)
		{
			((fc_layer*)output)->prepare(desired);
			output->learn();
		}
		void retrieve(double* where)
		{
			((fc_layer*)output)->unload(where);
		}
		void pack(std::string filename)
		{
			std::ofstream out(filename,std::ios::binary|std::ios::out);
			writeout(out,(unsigned int)layers.size());
			for(auto i:layers)
				i->pack(out);
			out.close();
		}
		~network()
		{
			for(layer* i:layers)
				delete i;
		}
		network(std::string filename)
		{
			std::ifstream inp(filename,std::ios::binary|std::ios::in);
			int i;
			readin(inp,&i);
			std::vector<int> d_n_s(3);
			for(;i>0;--i)
			{
				int layer_type;
				readin(inp,&layer_type);
				switch(layer_type)
				{
				case LAYER_ST:
					readin(inp,&(d_n_s[0]),3);
					input=new frontlayer(d_n_s);
					output=input;
					layers.push_back(output);
					break;
				case LAYER_FC:
					readin(inp,&(d_n_s[0]),3);
					output=new fc_layer(d_n_s,output,&inp);
					layers.push_back(output);
					break;
				case LAYER_CV:
					int f_d,f_n;
					readin(inp,&f_d);
					readin(inp,&f_n);
					output=new conv_layer(f_d,f_n,output,&inp);
					layers.push_back(output);
					break;
				case LAYER_RE:
					output=new relu_layer(output);
					layers.push_back(output);
					break;
				case LAYER_PL:
					output=new pool_layer(output);
					layers.push_back(output);
					break;
				default:
					exit(-1);
				}
			}
		}
		network(std::vector<int>&& parameters)
		{
			std::vector<int> d_n_s(3);
			for(unsigned int i=0;i<parameters.size();)
			{
				int layer_type=parameters[i];
				++i;
				switch(layer_type)
				{
				case LAYER_ST:
					for(int l=0;l<3;++l,++i)
						{
							d_n_s[l]=parameters[i];
						}
					input=new frontlayer(d_n_s);
					output=input;
					layers.push_back(output);
					break;
				case LAYER_FC:
					for(int l=0;l<3;++l,++i)
						d_n_s[l]=parameters[i];
					output=new fc_layer(d_n_s,output);
					layers.push_back(output);
					break;
				case LAYER_CV:
					int f_d,f_n;
					f_d=parameters[i];++i;
					f_n=parameters[i];++i;
					output=new conv_layer(f_d,f_n,output);
					layers.push_back(output);
					break;
				case LAYER_RE:
					output=new relu_layer(output);
					layers.push_back(output);
					break;
				case LAYER_PL:
					output=new pool_layer(output);
					layers.push_back(output);
					break;
				default:
					exit(-1);
				}
			}
		}
	};
}

void switch_endian(int* input)
{
	char* tmp=(char*)input;
	char temp=tmp[0];
	tmp[0]=tmp[3];
	tmp[3]=temp;
	temp=tmp[1];
	tmp[1]=tmp[2];
	tmp[2]=temp;
}

#define ubyte unsigned char

std::vector<double> get_label(std::ifstream& in)
{
	ubyte tmp;
	in.read((char*)&tmp,sizeof(tmp));
	std::vector<double> out(10,0.01);
	out[tmp]=0.09;
	return out;
}
std::vector<double> get_image(std::ifstream& in,int dimx,int dimy)
{
	std::vector<double> image(dimx*dimy,0);
	ubyte tmp;
	for(unsigned int i=0;i<image.size();++i)
	{
		in.read((char*)&tmp,sizeof(tmp));
		image[i]=tmp/255.;
	}
	return image;
}
ubyte from_arr_to_label(std::vector<double>& in)
{
	double max=-1;
	int ind=-1;
	for(unsigned int i=0;i<in.size();++i)
	{
		if(in[i]>max)
		{
			ind=i;max=in[i];
		}
	}
	return ind;
}

/* CNN from scratch:
	start with LAYER_ST supplied with 3 ints (dims)
	then any sequence of:
	LAYER_FC,3ints(dims)
	LAYER_CV,2ints(f_d,f_n)
	LAYER_RE
	LAYER_PL
	end with LAYER_FC and 3ints(dims)
*/	
//NOTICE: list.size() returns 64bit value (unsigned long long)
int main()
{
	std::ios_base::sync_with_stdio(0);
	nn::init();

	//MNIST reading
	std::ifstream input_label("./train-labels.idx1-ubyte",std::ios_base::binary);
	std::ifstream input_images("./train-images.idx3-ubyte",std::ios_base::binary);
	int32_t buffer;
	//reading magic numbers
	input_label.read((char*)&buffer,sizeof(buffer));
	input_images.read((char*)&buffer,sizeof(buffer));
	int32_t num,num2;
	input_label.read((char*)&num,sizeof(num));
	input_images.read((char*)&num2,sizeof(num2));
	assert(num==num2);
	switch_endian(&num);
	int32_t dimx,dimy;
	input_images.read((char*)&dimx,sizeof(dimx));
	input_images.read((char*)&dimy,sizeof(dimy));
	switch_endian(&dimx);
	switch_endian(&dimy);
	//Preps done
	//mnist_test.pack("cnn.packed");
	/*
	nn::network mnist_test("cnn.packed");
	std::vector<double> output=std::vector<double>(10);
	int hits=0;
	for(int i=0;i<num;++i)
	{
		if(!(i%100))
			std::cout<<"Starting sample "<<i+1<<" out of "<<num<<"; "<<hits*1.0/i<<std::endl;
		std::vector<double> image=get_image(input_images,dimx,dimy);
		mnist_test.launch(image);
		std::vector<double> label=get_label(input_label);
		mnist_test.retrieve(&output[0]);
		ubyte l1=from_arr_to_label(label);
		ubyte l2=from_arr_to_label(output);
		hits+=l1==l2;
		//std::cout<<(int)l1<<" -> "<<(int)l2<<"\n";
		std::cout<<output[0]<<std::endl;
		//mnist_test.train(&label[0]);
	}
	/*/
	//nn::network mnist_test("cnn.packed");
	std::vector<double> output=std::vector<double>(10);
	// nn::network mnist_test(std::vector<int>{LAYER_ST,dimx,dimy,1,LAYER_CV,3,3,LAYER_RE,LAYER_PL,LAYER_CV,3,3,LAYER_RE,LAYER_PL,LAYER_CV,3,3,LAYER_RE,LAYER_FC,10,1,1});
	// nn::network mnist_test(std::vector<int>{LAYER_ST,dimx,dimy,1,LAYER_FC,14,14,1,LAYER_FC,10,1,1});
	nn::network mnist_test(std::vector<int>{LAYER_ST,dimx,dimy,1,LAYER_FC,1,1,1,LAYER_FC,5,2,1});	
	for(int i=0;i<num;++i)
	{
		if(!(i%100))
			std::cout<<"Starting sample "<<i+1<<" out of "<<num<<std::endl;
		std::vector<double> image=get_image(input_images,dimx,dimy);
		mnist_test.launch(image);
		std::vector<double> label=get_label(input_label);
		mnist_test.retrieve(output.data());
		// for(double d:output)
		// 	std::cout<<d<<" ";
		// std::cout<<"\nBut want\n";
		// for(double d:label)
		// 	std::cout<<d<<" ";
		// std::cout<<"\n";
		mnist_test.train(&label[0]);
	}
	mnist_test.pack("cnn.packed");
	//*/
}