#pragma OPENCL EXTENSION cl_khr_fp64:enable
#define TS 32
#define TSDK 32
#define WPTM 2
#define WPTN 2
#define RTSM (TS/WPTM)
#define RTSN (TS/WPTN)
//matrix transpose
__kernel void transpose(const int m,
			const int n,
			__global const double* restrict input,
			int ldb,
			__global double* restrict output,
			int ldbt)
{
	int tx=get_local_id(0);
	int ty=get_local_id(1);
	int bx=get_group_id(0);
	int by=get_group_id(1);

	int BLOCK_SIZE=get_local_size(0);
	
	__local double buffer[16][17];
	
	int row=by*BLOCK_SIZE+ty;
	int col=bx*BLOCK_SIZE+tx;
	
	if(row<m && col<n)
		buffer[ty][tx]=input[row*ldb+col];
	barrier(CLK_LOCAL_MEM_FENCE);

	int newRow=bx*BLOCK_SIZE+ty;
	int newCol=by*BLOCK_SIZE+tx;

	if(newRow<n && newCol<m)
		output[newRow*ldbt+newCol]=buffer[tx][ty];
}

//vector norm
double impl_norm(
		__global const double* restrict vec,
		int start,
		int size,
		__local double* restrict tmp_buffer)
{
	double tmp=0;
	double vec_entry=0;
	for(unsigned int i=get_local_id(0); i<size; i+=get_local_size(0))
	{
		vec_entry=vec[i+start];
		tmp+=vec_entry*vec_entry;
	}
	tmp_buffer[get_local_id(0)]=tmp;
	for(unsigned int stride=get_local_size(0)/2; stride>0; stride/=2)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if(get_local_id(0)<stride)
			tmp_buffer[get_local_id(0)]+=tmp_buffer[get_local_id(0)+stride];
	}
	return tmp_buffer[0];
}

__kernel void norm(
		__global const double* restrict vec,
		int n,
		__global double* restrict group_buffer,
		__local double* restrict tmp_buffer)
{
	double tmp;
	tmp=impl_norm(vec, (get_group_id(0)*n)/get_num_groups(0), ((1+get_group_id(0))*n)/get_num_groups(0)-(get_group_id(0)*n)/get_num_groups(0), tmp_buffer);
	if(get_local_id(0)==0)
		group_buffer[get_group_id(0)]=tmp;
}

//matrix_vector multiplication
/*__kernel void mul_vec(
		int m,
		int n,
		__global const double4* restrict A,
		int lda,
		__global const double4* restrict x,
		__global double* restrict result,
		__local double4* restrict work)
{
	
	unsigned int row_gid=get_group_id(0);
	unsigned int col_gid=get_local_id(0);
	unsigned int lid=get_local_id(0);
	
	for(unsigned int row=row_gid;row<m;row+=get_num_groups(0))
	{
		double4 dot_prod={0.0,0.0,0.0,0.0};
		for(unsigned int col=col_gid; col<(n+3)/4; col+=get_local_size(0)){
			double4 Aval=A[row*(lda/4)+col];
			double4 xval=x[col];
			dot_prod+=Aval*xval;
		}
		work[lid]=dot_prod;
		barrier(CLK_LOCAL_MEM_FENCE);
		for(unsigned int stride=get_local_size(0)/2; stride>0; stride>>=1)
		{	
			if(lid<stride)
				work[lid]+=work[lid+stride];
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if(lid==0)
			result[row]=work[0].s0+work[0].s1+work[0].s2+work[0].s3;
	}
}*/
__kernel void mul_vec(int m,
		int n,
		__global const double* restrict A,
		int lda,
		__global const double* restrict x,
		__global double* restrict result,
		__local double* restrict work)
{
	
	unsigned int row_gid=get_group_id(0);
	unsigned int col_gid=get_local_id(0);
	unsigned int lid=get_local_id(0);
	
	for(unsigned int row=row_gid;row<m;row+=get_num_groups(0)) //coarse-grained parallelism
	{
		double dot_prod=0.0;
		for(unsigned int col=col_gid; col<n; col+=get_local_size(0)){ //fine-grained parallelism
			double Aval=A[row*lda+col];
			double xval=x[col];
			dot_prod+=Aval*xval;
		}
		work[lid]=dot_prod;
		barrier(CLK_LOCAL_MEM_FENCE);
		for(unsigned int stride=get_local_size(0)/2; stride>0; stride>>=1) //numerical merging method
		{	
			if(lid<stride)
				work[lid]+=work[lid+stride];
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if(lid==0)
			result[row]=work[0];
	}
}

//Some of the matrix line copy to another matrix
__kernel void matRowCopyMat(  
   			const int m,  
            const int n,  
            const int p,
			__global const int* restrict index,
            __global const double* restrict A, 
			int lda,  
            __global  double* restrict B,
			int ldb)
{
	unsigned int tid=get_local_id(0);
	unsigned int bid=get_group_id(0);
	for(unsigned int i=tid;i<n;i+=get_local_size(0))
                B[bid*ldb+i]=A[index[bid]*lda+i];
        barrier(CLK_GLOBAL_MEM_FENCE);
}
/*__kernel void matRowCopyMat(  
   			const int m,  
            const int n,  
            const int p,
			__global const int* restrict index,
            __global const double4* restrict A, 
			int lda,  
            __global  double4* restrict B,
			int ldb)
{
	unsigned int tid=get_local_id(0);
	unsigned int bid=get_group_id(0);
	for(unsigned int i=tid;i<(n+3)/4;i+=get_local_size(0))
                B[bid*(ldb/4)+i]=A[index[bid]*(lda/4)+i];
        barrier(CLK_GLOBAL_MEM_FENCE);
}*/

//Matrix MatrixTran multiplication
/*__kernel void MatrixMultMatrixTran(  
                        int m,  
                        int n,  
                        __global const double* restrict A, 
			int lda,  
                        __global double* restrict C,
			int ldc)
{
	int bx=get_group_id(0);
	int by=get_group_id(1);
	int tx=get_local_id(0);
	int ty=get_local_id(1);

	int row=by*TS+ty;
	int col=bx*TS+tx;

	__local double As[TS][TSDK];
	__local double Bs[TSDK][TS];
	
	double acc[WPT];
	for(int w=0; w<WPT; w++)
		acc[w]=0.0;

	for(int k=0; k<(n+TSDK-1)/TSDK; k++)
	{
		for(int l=0;l<WPT;l++){    //load one block of matrix A and B into local memory
			if(k*TSDK+tx>=n || by*TS+ty+l*RTS>=m ) 
				As[ty+l*RTS][tx]=0;
			else
			As[ty+l*RTS][tx]=A[(by*TS+ty+l*RTS)*lda+k*TSDK+tx];
			if(k*TSDK+tx>=n || bx*TS+ty+l*RTS>=m)
				Bs[tx][ty+l*RTS]=0;
			else
				Bs[tx][ty+l*RTS]=A[(bx*TS+ty+l*RTS)*lda+k*TSDK+tx];
			
			barrier(CLK_LOCAL_MEM_FENCE);
		}
			
		for(int unroll=0; unroll<TSDK; unroll++){ //calculate each point value of matrix C
			for(int w=0; w<WPT; w++)
				acc[w]+=As[ty+w*RTS][unroll]*Bs[unroll][tx];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
			
	}
	for(int w=0; w<WPT; w++)  //load data in private memory into global memory
		if(((row+w*RTS)<m) && (col<m))
			C[(row+w*RTS)*ldc+col]=acc[w];
}*/
__kernel void MatrixMultMatrixTran(  
                        int m,  
                        int n,  
                        __global const double* restrict A, 
			            int lda,  
                        __global double* restrict C,
			            int ldc)
{
	int bx=get_group_id(0);
	int by=get_group_id(1);
	int tx=get_local_id(0);
	int ty=get_local_id(1);

	int row=by*TS+ty;
	int col=bx*TS+tx;

	__local double As[TS][TSDK];
	__local double Bs[TSDK][TS];
	
	double Areg;
	double Breg[WPTN];
	double acc[WPTM][WPTN];

	for(int wm=0; wm<WPTM; wm++){
		//#pragma unroll
		for(int wn=0;wn<WPTN;wn++){
			acc[wm][wn]=0.0;
		}
	}
	for(int k=0; k<(n+TSDK-1)/TSDK; k++)
	{
		for(int l=0; l<WPTM; l++)  //load one block of matrix A and B into local memory
			for(int w=0; w<WPTN; w++){
				if(k*TSDK+tx+w*RTSN>=n || by*TS+ty+l*RTSM>=m)
					As[ty+l*RTSM][tx+w*RTSN]=0;
				else
					As[ty+l*RTSM][tx+w*RTSN]=A[(by*TS+ty+l*RTSM)*lda+k*TSDK+tx+w*RTSN];
				if(k*TSDK+tx+w*RTSN>=n || bx*TS+ty+l*RTSM>=m)
					Bs[tx+w*RTSN][ty+l*RTSM]=0;
				else
					Bs[tx+w*RTSN][ty+l*RTSM]=A[(bx*TS+ty+l*RTSM)*lda+k*TSDK+tx+w*RTSN];
			}
			
		barrier(CLK_LOCAL_MEM_FENCE);
		for(int t=0;t<TSDK;t++){    //calculate each point value of matrix C
			for(int wm=0;wm<WPTM;wm++){
				int row1=ty+wm*RTSM;
				Areg=As[row1][t];
				for(int wn=0;wn<WPTN;wn++){
					int col1=tx+wn*RTSN;
					Breg[wn]=Bs[t][col1];
				acc[wm][wn]+=Areg*Breg[wn];
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
			
	}
	for(int wm=0; wm<WPTM; wm++)  //load data in private memory into global memory
		for(int wn=0; wn<WPTN; wn++)
			if( ((row+wm*RTSM)<m) && ((col+wn*RTSN)<m) )
				C[(row+wm*RTSM)*ldc+col+wn*RTSN]=acc[wm][wn];
}

//solve trigonometric function 
__kernel void triangular_substitute_inplace(
			int m,
			int n,
			__global const double* restrict A,
			int lda,
			__global double* restrict x,
			 int options)
{	
	double temp;
	unsigned int unit_diagonal_flag=(options & (1<<0));
	unsigned int transposed_access_A = (options & (1<<1));
	unsigned int is_lower_solve = (options & (1<<2));
	unsigned int row;
	for(unsigned int rows_processed=0; rows_processed<m; ++rows_processed)
	{
		row=is_lower_solve ? rows_processed: ((m-rows_processed)-1);
		if(!unit_diagonal_flag)
		{
			barrier(CLK_GLOBAL_MEM_FENCE);
			if(get_global_id(0)==0)
				x[row]/=A[row*lda+row];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
		temp=x[row];

		for(int elim=(is_lower_solve ? (row+get_global_id(0)+1):get_global_id(0)); elim<(is_lower_solve ? m:row ); elim+=get_global_size(0))
			x[elim]-=temp*A[transposed_access_A?(row*lda+elim):(elim*lda+row)];
		barrier(CLK_GLOBAL_MEM_FENCE);
	
	}
}

//vector subtract
/*__kernel void  vectorSub_kernel(
			__global const double4 *x,
			__global const double4 *y,
			const int n,
			__global double4 *result)
{

	for(unsigned int i=get_global_id(0); i<(n+3)/4; i+=get_global_size(0))
		result[i]=x[i]-y[i];
}*/

__kernel void  vectorSub_kernel(
			__global const double* restrict	x,
			__global const double* restrict y,
			const int n,
			__global double* restrict result)
{

	for(unsigned int i=get_global_id(0); i<n; i+=get_global_size(0))
		result[i]=x[i]-y[i];
}

