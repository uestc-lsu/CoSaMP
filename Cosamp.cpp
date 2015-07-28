#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <fstream>
#include <string.h>
#include <time.h>
#include "head.h"
#include <iostream>
using namespace std;

//y=A*x
void MattranMultVec(double *A,double *x,double *y,int row,int col)
{
   for(int i=0;i<col;i++){
	   y[i] =0;
	   for(int j=0; j<row;j++)
		   y[i] =y[i] +A[j*col+i]*x[j] ;
   }
} 
//C=A*A'
void MatMultMattran(double *A,double *C,int m,int n)
{
        int i,j,k;
        for(i=0;i<m;i++){
                for(j=i;j<m;j++){
                        C[i*m+j]=0;
                        for(k=0;k<n;k++){
                                C[i*m+j]+=A[i*n+k]*A[j*n+k];

                        }
                        if(i!=j)
                                C[j*m+i]=C[i*m+j];
                }
        }
} 

void MatTran(double *A, double *B,int m,int n)
{  for(int i=0;i<m;i++)
     for(int j=0;j<n;j++)
		 B[j*m+i]=A[i*n+j];
 }

void MatMultVec(double *A,double *x,double *y,int row,int col)
{
   for(int i=0;i<row;i++){
	   y[i]=0;
	   for(int j=0; j<col;j++)
		   y[i]=y[i]+A[i*col+j]*x[j];
   }
}

int VecIndex(double *x,int m)
{ 
	int index;
    double max=0;
    for(int i=0;i<m;i++){
		x[i]=fabs(x[i]);
		if(x[i]>max){
		max=x[i];
        index=i;
		}
    }
  return index;
}

void MatColCopyMat(double *A,double *B,int *index,int num,int m,int n)
{
	int i,j;
	for(i=0;i<num;i++)
		for(j=0;j<n;j++)
			B[n*i+j]=A[index[i]*n+j];
}

double Norm(double *x,int width)
{
	double temp=0;
	int i;
	for(i=0;i<width;i++){
		temp = temp +  x[i] * x[i];
	}
	temp = sqrt(temp);
	return  temp;
}

int  Union(int *A,int elem,int num)
{
	int i,j,temp;
	if (num==0){
		A[0] = elem;
		return 1;
	}
	for(i=0;i<num;i++){   
		if (A[i]==elem)
			return 0;
		else if (A[i]>elem){
			temp = A[i];
			A[i] = elem;
			for (j=num;j>i;j--){
				A[j] = A[j-1];
			}
			A[i+1] = temp;
			return 1;
		}
	}
	A[num] = elem;
	return 1;
}

void subVec(const double *op1,const double *op2,double *result,int size)
{
	int i;
	for(i = 0;i < size; i++){
		result[i] = op1[i];
		result[i]=result[i] - op2[i];
	}
}

void copyVec(double *desc,int dInc,const double *src,int sInc,int size)
{
	int i;
	for(i = 0;i < size; i++){
		memcpy(desc+i * dInc,src +i * sInc,sizeof(double));
	}
}

void MatCol(double *A,double *a,int m,int n,int index)
{
	for(int i=0;i<m;i++)
	  a[i]=A[i*n+index];
}

int cholesky(double a[],int n)
{
	int i,j,k,u,v;
	if ((a[0]+1.0==1.0)||(a[0]<0.0))
	{ 
		cout<<"fail"<<endl;
		return(false);
	}
	a[0]=sqrt(a[0]);
	for (j=1; j<=n-1; j++) a[j]=a[j]/a[0];
	for (i=1; i<=n-1; i++)
	{
		u=i*n+i;
		for (j=1; j<=i; j++)
		{ 
			v=(j-1)*n+i;
			a[u]=a[u]-a[v]*a[v];
		}
		if ((a[u]+1.0==1.0)||(a[u]<0.0))
		{ 
			cout<<"fail"<<endl;
			return(false);
		}
		a[u]=sqrt(a[u]);
		if (i!=(n-1))
		{ 
			for (j=i+1; j<=n-1; j++)
			{ 
				v=i*n+j;
				for (k=1; k<=i; k++)
					a[v]=a[v]-a[(k-1)*n+i]*a[(k-1)*n+j];
					a[v]=a[v]/a[u];
			}
		}
	}

}
int chlk(double a[], int n, int m, double d[])
{
	int i,j,k,u,v;
	for (j=0; j<=m-1; j++)
	{ 
		d[j]=d[j]/a[0];
		for (i=1; i<=n-1; i++)
		{ 
			u=i*n+i; 
			v=i*m+j;
			for (k=1; k<=i; k++)
				d[v]=d[v]-a[(k-1)*n+i]*d[(k-1)*m+j];
				d[v]=d[v]/a[u];
		}
	}
	for (j=0; j<=m-1; j++)
	{
		u=(n-1)*m+j;
		d[u]=d[u]/a[n*n-1];
		for (k=n-1; k>=1; k--)
		{ 
			u=(k-1)*m+j;
			for (i=k; i<=n-1; i++)
			{ 
				v=(k-1)*n+i;
				d[u]=d[u]-a[v]*d[i*m+j];
			}
			v=(k-1)*n+k-1;
			d[u]=d[u]/a[v];
		}
	}
	return(true);
}
//Least Square Method,use cholesky decompose
void LeastSquare(double *A,double *x,int m,int n,double *y)
{ 
    	int i,j;

	double *B = new double [m*m];
	double *b = new double [m*1];
	
	MatMultMattran(A,B,m,n);
	MatMultVec(A,x,b,m,n);
	cholesky(B,m);
	int flag=chlk(B, m,  1, b);
	if (!flag)
		cout<<"Error!"<<endl;
	else
		for (i=0; i<m; i++)
			y[i] = (float)b[i];
	delete []B;
	delete []b;
}
//The absolute value of the one-dimensional array element is from the big to small sort, sorting the results for the order of the foot
void sort(double *a,int *index,int num,int n)
{
	int i,j,x,*flag;
	double max,temp,*a_new;
	flag=new int[n];
	a_new=new double[n];
	for (i=0;i<n;i++)
	{
		a_new[i] = fabs(a[i]);
		flag[i] = 1;
	}
	max = a_new[0];
	x = 0;
	for (i=0;i<num;i++)
	{
		for (j=0;j<n;j++)
		{
			if ( flag[j]==0 )
				continue;
			temp = a_new[j];
			if ( temp >= max )
			{
				max = temp;
				x = j;
			}
		}
		flag[x] = 0; 
		index[i] = x;
		max = 0;
	}
	delete []flag;
	delete []a_new;
}

void BubbleSort(int *a,int num)
{
	int i,j,temp;
	for (i=0;i<num-1;i++)
	{
		for (j=num-1;j>i;j--)
		{
			if (a[j-1]>a[j])
			{
				temp = a[j-1];
				a[j-1] = a[j];
				a[j] = temp;
			}
		}
	}
}
//If there is a repeated element check array, any array is removed. According to the order from small to large.
int Check(int *a,int num)
{
	int i,j,result;
	result = 0;		
	for (i=0;i<num-1;i++)
	{
		if (a[i]==a[i+1])
		{
			for (j=i+1;j<num-1-result;j++)
				a[j] = a[j+1];
			a[num-1-result] = -1-result;
			result++;
		}
	}
	return result;
}

void cosamp(double* measure, double* recSignal, double* A, int m, int n, double err,int k )
{
	double *rError,*h,*temp1,*temp2,*temp3,*temp4,*temp5,*temp6,*subMat_A,eps,*temp3_temp;
	int num,ka,*support,alpha,*h_index,*a_index,*a_index2,i,support_num,result,sort_num;
	result = 0;
	alpha = 2;
	support_num = k + alpha*k;
	support=new int[n];
	h_index=new int[n];
	a_index=new int[n];
	a_index2=new int[k];
	rError=new double[m];
	subMat_A=new double[n*m];
	h=new double[n];
	temp2=new double[m];
	temp1= new double[n];

	for (i=0;i<n;i++)
		recSignal[i] = 0.0;

	copyVec(rError,1,measure,1,m);
	num = 0;
	while ( num<k &&Norm(rError,m)>err )
	{
		//calculate signal agent
		MatMultVec(A,rError,h,n,m);
		// expand support set
		if ( num == 0 )
		{
			sort_num=alpha*k;
			sort(h,h_index,sort_num,n);
			for(i=0;i<sort_num;i++)
				support[i]=h_index[i];
			support_num = alpha*k;
			BubbleSort(support,alpha*k);
		}
		else
		{
			for (i=0;i<k;i++)
				support[i] = a_index2[i];
			sort_num=alpha*k;
			sort(h,h_index,sort_num,n);
			for (i=0;i<sort_num;i++)
				support[k+i] = h_index[i];
			support_num = k + alpha*k;
			BubbleSort(support,(alpha+1)*k);
		}
		result = Check(support,support_num);
		support_num = support_num - result;
		num++;
		//the result of least square method
		temp3=new double[support_num *m];	
		//Some of the matrix line copy to another matrix
		MatColCopyMat(A,temp3,support,support_num ,n,m);
		//estimation signal,Least Square method
		LeastSquare(temp3,measure,support_num,m,temp1);
		//prune signal
		sort(temp1,a_index,k,support_num);
		memset(recSignal,0,sizeof(double)*n);
		for(i=0;i<k;i++)
			recSignal[support[a_index[i]]]=temp1[a_index[i]];
		for(i=0;i<k;i++)
			a_index2[i] = support[a_index[i]];
		//update the residual
		MattranMultVec(A,recSignal ,temp2, n,m);
		subVec(measure,temp2,rError,m);	
		delete []temp3;
	}
	delete []temp1;
	delete []support;
	delete []h_index;
	delete []a_index;
	delete []a_index2;
	delete []rError;
	delete []subMat_A;
	delete []h;
	delete []temp2;

}

void CoSaMPMat(double *SampleMat,double *GaussMat,int bmpWidth,int bmpHeight,int m,double *RecMat)
{
	double *ColSignal,*ColRecSig,*A;

	int i,j,k,index;
	k=m/4;
	ColSignal=(double *)malloc( m* sizeof(double));
	ColRecSig=(double *)malloc( bmpHeight* sizeof(double));
	A=(double *)malloc(bmpHeight*m*sizeof(double));
	MatTran(GaussMat,A,m,bmpHeight);
	for(i=0;i<bmpWidth;i++){ 
		index=i;
		memset(ColSignal,0,sizeof(double)*m);
		memset(ColRecSig,0,sizeof(double)*bmpHeight);
		MatCol(SampleMat,ColSignal,m,bmpWidth,index);
		cosamp(ColSignal,ColRecSig,A,m,bmpHeight,0.001,k);
		for( j=0;j<bmpHeight;j++){
			RecMat[j*bmpWidth+i]=ColRecSig[j];
		}
	}	
	delete []ColSignal;
	delete []ColRecSig;

}
