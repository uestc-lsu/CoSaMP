
#include<fstream>
#include <stdio.h>
#include <CL/cl.h>
#include<string>
using namespace std;

#ifndef HEADER
#define HEADER
bool readImageGDAL(unsigned char **pImageData,int &width,int &height,int &nChannels, const char *filePath,double trans[6],char pszSrcWKT[]);
char *strlwr(char *s);
char* findImageTypeGDAL( char *pDstImgFileName);
bool WriteImageGDAL( char* pDstImgFileName,unsigned char * pImageData,int width,int height,int nChannels,double trans[6],char *pszSrcWKT);

void filterset(double t);
void IDwt1D(double *buffer, int buflen);
void Dwt1D(double *buffer, int buflen);
void IDwtND(double buffer[], int height, int width, int lv);
void DwtND(double buffer[], int height, int width, int lv);
void Adjust(double *ImageData,int bmpHeight,int bmpWidth, double *SparseMat,short *OutData);

void grns(double *A,int M,int N);
double uniform(double a, double b, long int *seed);
double gauss(double mean, double sigma, long int * s) ;
void CreateSampleMat(double *A,int M,int N);
void MatMul(double *A,double *B,double *C,int a,int b,int c);
void  CompressSample(double *SparseMat,int m,int bmpWidth,int bmpHeight,double *GaussMat,double *SampleMat);

void MattranMultMat(double *A,double *C,int m,int n);
void MattranMultVec1(double *A,double *x,double *y,int row,int col);
void MattranMultVec(double *A,double *x,double *y,int row,int col);
void MatMultMattran(double *A,double *C,int m,int n);
void Matrix_Inverse(double *Coefficient_Matrix_K,int num) ;
void MatMult(double *A,double *B,double *C,int n,int m,int x);
void MatTran(double *A, double *B,int m,int n);
void MatMultVec(double *A,double *x,double *y,int row,int col);
int VecIndex(double *x,int m);
void MatColCopyMat(double *A,double *B,int *index,int num,int m,int n);
double Norm(double *x,int width);
int  Union(int *A,int elem,int num);
void subVec(const double *op1,const double *op2,double *result,int size);
void copyVec(double *desc,int dInc,const double *src,int sInc,int size);
void MatCol(double *A,double *a,int m,int n,int index);
int cholesky(double a[],int n);
int chlk(double a[], int n, int m, double d[]);
void LeastSquare(double *A,double *x,int m,int n,double *y);
void sort(double *a,int *index,int num,int n);
void BubbleSort(int *a,int num);
int Check(int *a,int num);
void cosamp(double* measure, double* recSignal, double* A, int m, int n, double err,int k );
void CoSaMPMat(double *SampleMat,double *GaussMat,int bmpWidth,int bmpHeight,int m,double *RecMat);//on CPU
void ck(cl_int ret,const char *mes);
bool GetFileData(const char *path,string & str);
extern "C" void setMatrix_async(int m,int n,double const* hA_src,int ldha,cl_mem dB_dst,size_t dB_offset,int lddb,cl_command_queue queue,cl_event *event);
extern "C" void getMatrix_asyc(int m,int n,cl_mem dA_src,size_t dA_offset,int ldda,double *hB_dst,int ldhb,cl_command_queue queue,cl_event *event);
void MatrixMulGPU(double *A,double *B,double *C,int m,int n,int p);
void CosampMat(double *SampleMat,double *GaussMat,int bmpWidth,int bmpHeight,int m,double *RecMat);//on GPU
#endif
