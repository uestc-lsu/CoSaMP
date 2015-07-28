//#include "stdafx.h"
#include <iostream>
#include <time.h>
#include <ctime>
#include <cstdlib>
#include "math.h"
#include "head.h"
using namespace std;

double gaussrand()
{
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if ( phase == 0 ) {
		do {
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while(S >= 1 || S == 0);

	X = V1 * sqrt(-2 * log(S) / S);
	}
	 else
	X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

return X;
} 
//create sample matrix
void CreateSampleMat(double *A,int M,int N)
{
	int i, j;
	for(i = 0; i < M; i++)
	{
		for(j = 0; j <N; j++)
		{
			A[i*N+j] = gaussrand();
			
		}
			
	}
	
}

void MatMult(double *A,double *B,double *C,int a,int b,int c)
{ int i,j,k;
	for(i=0;i<a;i++)
	{
		for(j=0;j<b;j++)
		{  C[i*b+j]=0;
			for(k=0;k<c;k++)
			{
				C[i*b+j]+=A[i*c+k]*B[k*b+j];
			}	
		}
	}
}
//image sample
void  CompressSample(double *SparseMat,int m,int width,int height,double *GaussMat,double *SampleMat)
{
  	CreateSampleMat(GaussMat,m,height);
	MatrixMulGPU(GaussMat,SparseMat,SampleMat,m,width,height);
   	//MatMult(GaussMat,SparseMat,SampleMat,m,width,height);

}
