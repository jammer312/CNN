__kernel void add_kernel(__global double* other,__global double* out,int2 size)
{
	int j=get_global_id(0);
	int k=get_global_id(1);
	out[j+k*size.s0]+=other[j+k*size.s0];
}
__kernel void sub_kernel(__global double* other,__global double* out,int2 size)
{
	int j=get_global_id(0);
	int k=get_global_id(1);
	out[j+k*size.s0]-=other[j+k*size.s0];
}
__kernel void add_number_kernel(__global double* out,double adder,int2 size)
{
	int j=get_global_id(0);
	int k=get_global_id(1);
	out[j+k*size.s0]+=adder;	
}
__kernel void mul_kernel(__global double* in,__global double* other,__global double* out,int2 size1,int2 size2,int2 size3)
{
	//size -> rowsXcols
	int j=get_global_id(0);//rows
	int k=get_global_id(1);//cols
	out[j*size3.s1+k]=0;
	for(int i=0;i<size1.s1;++i)
	{
		out[j*size3.s1+k]+=in[j*size1.s1+i]*other[i*size2.s1+k];
	}
}
__kernel void mul_number_kernel(__global double* out,double multiplier,int2 size)
{
	int j=get_global_id(0);
	int k=get_global_id(1);
	out[j+k*size.s0]*=multiplier;
}
__kernel void halberd_product_kernel(__global double* in,__global double* other,int2 size)
{
	int j=get_global_id(0);
	int k=get_global_id(1);
	in[j+k*size.s0]*=other[j+k*size.s0];
}
__kernel void transpone_kernel(__global double* in,__global double* out,int2 size)
{
	int j=get_global_id(0);
	int k=get_global_id(1);
	out[j+k*size.s0]=in[k+j*size.s1];
}
double sigmoid(double in)
{
	return 1.0/(1.0+exp(-in));
}
double sigmoid_prime(double in)
{
	return sigmoid(in)*(1-sigmoid(in));
}
__kernel void sigmoid_kernel(__global double* in,__global double* out,int2 size)
{
	int j=get_global_id(0);
	int k=get_global_id(1);
	out[j+k*size.s0]=sigmoid(in[j+k*size.s0]);
}
__kernel void sigmoid_prime_kernel(__global double* in,__global double* out,int2 size)
{
	int j=get_global_id(0);
	int k=get_global_id(1);
	out[j+k*size.s0]=sigmoid(in[j+k*size.s0])*(1-sigmoid(in[j+k*size.s0]));
}
__kernel void tanh_kernel(__global double* in,__global double* out,int2 size)
{
	int j=get_global_id(0);
	int k=get_global_id(1);
	out[j+k*size.s0]=tanh(in[j+k*size.s0]);
}
__kernel void tanh_prime_kernel(__global double* in,__global double* out,int2 size)
{
	int j=get_global_id(0);
	int k=get_global_id(1);
	double tmp=tanh(in[j+k*size.s0]);
	out[j+k*size.s0]=1-tmp*tmp;
}