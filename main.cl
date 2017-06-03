double relu(double x)
{
	return max(.0,x);
}
double relu_d(double x)
{
	return x>0?1:0;
}
double sigmoid(double in)
{
	return 1.0/(1.0+exp(-in));
}
double sigmoid_prime(double in)
{
	return sigmoid(in)*(1-sigmoid(in));
}
//read with padding
double read(__global double* array,int3 size,int3 point)
{
	if(point.s0>=0&&point.s1>=0&&point.s2>=0&&point.s0<=size.s0&&point.s1<=size.s1&&point.s2<=size.s2)
		return array[point.s0+point.s1*size.s0+point.s2*size.s1*size.s0];
	return 0;
}
//unsafe with offset for fc
double specialread(__global double* array,int3 size,int3 point,int3 ext_size,int3 ext_point)
{
	int smallsize=size.s0*size.s1*size.s2;
	return array[smallsize*(ext_point.s0+ext_point.s1*ext_size.s0+ext_point.s2*ext_size.s1*ext_size.s0)+point.s0+point.s1*size.s0+point.s2*size.s1*size.s0];
}
void specialmodify(__global double* array,int3 size,int3 point,int3 ext_size,int3 ext_point,double modificator)
{
	int smallsize=size.s0*size.s1*size.s2;
	array[smallsize*(ext_point.s0+ext_point.s1*ext_size.s0+ext_point.s2*ext_size.s1*ext_size.s0)+point.s0+point.s1*size.s0+point.s2*size.s1*size.s0]+=modificator;
}
//unsafe for convolution
double convspecialread(__global double* array,int3 size,int3 point,int3 ext_size,int3 ext_point)
{
	int smallsize=size.s0*size.s1*size.s2;
	return array[smallsize*(ext_point.s2)+point.s0+point.s1*size.s0+point.s2*size.s1*size.s0];
}
void convspecialmodify(__global double* array,int3 size,int3 point,int3 ext_size,int3 ext_point,double modificator)
{
	int smallsize=size.s0*size.s1*size.s2;
	 array[smallsize*(ext_point.s2)+point.s0+point.s1*size.s0+point.s2*size.s1*size.s0]+=modificator;
}
//unsafe write
void write(__global double* array,int3 size,int3 point,double value)
{
	array[point.s0+point.s1*size.s0+point.s2*size.s1*size.s0]=value;
}
void write_add(__global double* array,int3 size,int3 point,double value)
{
	array[point.s0+point.s1*size.s0+point.s2*size.s1*size.s0]+=value;
}
__kernel void fc_feedforward_kernel(__global double* in,__global double* out,__global double* weighted,__global double* biases,__global double* weights,int3 size,int3 size_prev)
{
	int i=get_global_id(0);
	int j=get_global_id(1);
	int k=get_global_id(2);
	int3 point = (int3)(i,j,k);
	double tmpvalue=read(biases,size,point);
	for(int x=0;x<size_prev.s0;++x)
		for(int y=0;y<size_prev.s1;++y)
			for(int z=0;z<size_prev.s2;++z)
				tmpvalue+=specialread(weights,size_prev,(int3)(x,y,z),size,point)*read(in,size_prev,(int3)(x,y,z));
	write(weighted,size,point,tmpvalue);
	tmpvalue=sigmoid(tmpvalue);
	write(out,size,point,tmpvalue);
}

__kernel void fc_internalize_error_kernel(__global double* error,__global double* weighted,int3 size)
{
	int i=get_global_id(0);
	int j=get_global_id(1);
	int k=get_global_id(2);
	int3 point = (int3)(i,j,k);
	write(error,size,point,read(error,size,point)*sigmoid_prime(read(weighted,size,point)));
}

//tricky way, but kinda best out there
//we launch from previous layer error field and try to compute errors given existing internal error for current layer
__kernel void fc_backpropagate_kernel(__global double* out_error,__global double* cur_error,__global double* weights,
	int3 size_prev,int3 size)
{
	int i=get_global_id(0);
	int j=get_global_id(1);
	int k=get_global_id(2);
	int3 point = (int3)(i,j,k);//point in prevlayer
	double tmpvalue=0; //here will we store error
	for(int x=0;x<size.s0;++x)
		for(int y=0;y<size.s1;++y)
			for(int z=0;z<size.s2;++z)
				tmpvalue+=specialread(weights,size_prev,point,size,(int3)(x,y,z))*read(cur_error,size,(int3)(x,y,z));
	write(out_error,size_prev,point,tmpvalue);
	//Done. All we do here is move error from fc layer to some other
}
__kernel void fc_adapt_kernel(__global double* error,__global double* prev_activation,
 __global double* weights, __global double* biases,int3 size,int3 size_prev,double learnrate)
{
	int i=get_global_id(0);
	int j=get_global_id(1);
	int k=get_global_id(2);
	int3 point = (int3)(i,j,k);
	write_add(biases,size,point,learnrate*read(error,size,point));
	for(int x=0;x<size_prev.s0;++x)
		for(int y=0;y<size_prev.s1;++y)
			for(int z=0;z<size_prev.s2;++z)
				specialmodify(weights,size_prev,(int3)(x,y,z),size,point,-learnrate*read(error,size_prev,(int3)(x,y,z))*read(prev_activation,size_prev,(int3)(x,y,z)));
}
__kernel void cv_feedforward_kernel(__global double* in, __global double* out,__global double* filters,
	int fsize,int3 size,int3 size_prev)
{
	int i=get_global_id(0);
	int j=get_global_id(1);
	int k=get_global_id(2);
	int3 point = (int3)(i,j,k);
	int halfsize=fsize/2;
	int3 filtersize=(int3)(fsize,fsize,size_prev.s2);
	double tmpvalue=0;
	for(int x=i-halfsize;x<=i+halfsize;++x)
		for(int y=i-halfsize;y<=i+halfsize;++y)
			for(int z=0;z<size_prev.s2;++z)
			{
				tmpvalue+=read(in,size_prev,(int3)(x,y,z))*(convspecialread(filters,filtersize,(int3)(x-i+halfsize,y-i+halfsize,z),size,point));
			}
	write(out,size,point,tmpvalue);
}
//same here, all we want is to transfer error from current to previous layer, and we do it by launching from previous layer
//no need to internalize error for convolution cause it's same as external
__kernel void cv_backpropagate_kernel(__global double* out_error,__global double* cur_error,__global double* filters,
	int fsize,int3 size_prev,int3 size)
{
	int i=get_global_id(0);
	int j=get_global_id(1);
	int k=get_global_id(2);
	int3 point = (int3)(i,j,k); //point in prev layer
	int halfsize=fsize/2;
	int3 filtersize=(int3)(fsize,fsize,size_prev.s2);
	double tmpvalue=0;
	for(int x=-halfsize;x<=halfsize;++x)
		for(int y=-halfsize;y<=halfsize;++y)
			for(int z=0;z<size.s2;++z)
			{
				int3 curp=(int3)(point.s0+x,point.s1+y,z);
				int3 curwf=(int3)(-x,-y,point.s2);
				tmpvalue+=convspecialread(filters,filtersize,curwf,size,curp)*read(cur_error,size,curp);
			}
	write(out_error,size_prev,point,tmpvalue);
}
//Some advanced trickery here
//Now we launch straight from filters and sum deltas all over the layer
//And also it's 4d now
// Whoops, no 4d for now, falling back to 3d
__kernel void cv_adapt_kernel(__global double* fweights,__global double* cur_error,__global double* prev_activation,
	int3 size,int3 size_prev, int fsize, double learnrate)
{
	int a=get_global_id(0);//x
	int b=get_global_id(1);//y
	int c=get_global_id(2);//z
//	int d=get_global_id(3);//filter
	int3 point = (int3)(a,b,c);
	int3 filtersize=(int3)(fsize,fsize,size_prev.s2);
	int halfsize=fsize/2;
	int relx=-halfsize+a;
	int rely=-halfsize+b;
	double tmpdelta=0;
	for(int d=0;d<size.s2;++d)
	{		
		for(int x=0;x<size.s0;++x)
			for(int y=0;y<size.s1;++y)
			{
				int3 curp=(int3)(x+relx,y+rely,c);
				tmpdelta+=read(prev_activation,size_prev,curp)*read(cur_error,size,(int3)(x,y,d));
			}
		tmpdelta/=size.s0*size.s1;
		fweights[a+b*fsize+c*fsize*fsize+d*fsize*fsize*size_prev.s2]-=tmpdelta*learnrate;
	}
}
__kernel void relu_feedforward_kernel(__global double* in,__global double* out,int3 size)
{
	int i=get_global_id(0);
	int j=get_global_id(1);
	int k=get_global_id(2);
	int3 point = (int3)(i,j,k);
	write(out,size,point,relu(read(in,size,point)));
}
__kernel void pool_feedforward_kernel(__global double* in, __global double* out, int3 size, int dim)
{
	int i=get_global_id(0);
	int j=get_global_id(1);
	int k=get_global_id(2);
	int3 point = (int3)(i,j,k);
	double _max=0; //apply to RELUed data
	for(int _i=0;_i<dim;++_i)
		for(int _j=0;_j<dim;++_j)
		{
			_max=max(_max,read(in,size,(int3)(i*dim+_i,j*dim+_j,k)));
		}
	write(out,size,point,_max);
}
//at least these seem easy
__kernel void relu_backpropagate_kernel(__global double* out_error,__global double* cur_error,__global double* prev_activation,int3 size)
{
	int i=get_global_id(0);
	int j=get_global_id(1);
	int k=get_global_id(2);
	int3 point = (int3)(i,j,k);
	write(out_error,size,point,read(cur_error,size,point)*relu_d(read(prev_activation,size,point)));
}
//Notice: size for current layer, not the one we launch from
__kernel void pool_backpropagate_kernel(__global double* out_error,__global double* cur_error,__global double* prev_activation, __global double* cur_activation,int3 cur_size,int dim)
{
	int i=get_global_id(0);
	int j=get_global_id(1);
	int k=get_global_id(2);
	int3 point = (int3)(i,j,k);//in prev layer
	int3 cur_point=point/dim;
	double activator=read(cur_activation,cur_size,cur_point);
	write(out_error,cur_size*3,point,activator==read(prev_activation,cur_size*3,point)?read(cur_error,cur_size,cur_point):0);
}
__kernel void initial_error_calc_kernel(__global double* activation, __global double* desired,__global double* error,__global double* weighted)
{
	int i=get_global_id(0);
	error[i]=(activation[i]-desired[i])*sigmoid_prime(weighted[i]);
}