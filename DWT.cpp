
#include "math.h"
#include <iostream>
#include "head.h"
//using namespace std;

int shift=4;
double *h, *h1;
double p_alfa, p_beta, p_gama, p_delta, p_kesa,p_t=0.730174 ;


void filterset(double t)
{
	h = new double [9];
	h1 = new double [7];
	*h = (8*t*t*t-6*t*t+3*t)/(1+2*t)*(1/32.)*sqrt(2.0);
	*(h+1) = (-16*t*t*t+20*t*t-12*t+3)/(1+2*t)*(1/32.)*sqrt(2.0);
	*(h+2) = (2*t-3)/(1+2*t)*(1/8.)*sqrt(2.0);
	*(h+3) = (16*t*t*t-20*t*t+28*t+5)/(1+2*t)*(1/32.)*sqrt(2.0);
	*(h+4) = (-8*t*t*t+6*t*t+5*t+20)/(1+2*t)*(1/16.)*sqrt(2.0);
	*(h+5) = *(h+3);
	*(h+6) = *(h+2);
	*(h+7) = *(h+1);
	*(h+8) = *(h+0);

	double r0, r1, s0, t0;
	r0 = (*(h+4))-2*(*h)*(*(h+3))/(*(h+1));
	r1 = (*(h+2))-(*h)-(*h)*(*(h+3))/(*(h+1));
	s0 = (*(h+3))-(*(h+1))-(*(h+1))*r0/r1;
	t0 = r0-2*r1;

	p_alfa = (*h)/(*(h+1));
	p_beta = (*(h+1))/r1;
	p_gama = r1/s0;
	p_delta = s0/t0;
	p_kesa = t0;


}

void Dwt1D(double *buffer, int buflen)
{
	int i;
	int itemp;

	double *d, *s, *p;
	p = new double [buflen+2*shift];
	d = new double [(buflen>>1)+shift];
	s = new double [(buflen>>1)+shift];

	for (i=0; i<(buflen>>1)+shift; i++){
		itemp = i-(shift>>1);
		*(d+i) = *(p+shift+2*itemp+1) + p_alfa*( *(p+shift+2*itemp)+*(p+shift+2*itemp+2) );
	}

	for (i=0; i<(buflen>>1)+shift-1; i++){
		itemp = i+1-(shift>>1);
		*(s+i+1) = *(p+shift+2*itemp) + p_beta*( *(d+i+1)+*(d+i+1-1) );
	}
  
	for (i=0; i<(buflen>>1)+shift-1; i++){
		//itemp = i-(shift>>1);
		*(p+shift+(buflen>>1)+i) =  *(d+i) + p_gama*( *(s+i)+*(s+i+1)); 
	}
	/*  s2 */
    for (i=0; i<(buflen>>1)+shift-1; i++){
	//	itemp = i-(shift>>1);
		*(p+i+1) = *(s+i+1) + p_delta*(*(p+(buflen>>1)+shift+i+1) + *(p+(buflen>>1)+shift+i+1-1) );
	}

	/* d3 */
	for (i=0; i<(buflen>>1); i++)
		*(d+i) = *(p+i+(buflen>>1)+shift+(shift>>1)) +p_kesa*(1-p_kesa)*(*(p+i+(shift>>1)));
	/* s3 */
	for (i=0; i<(buflen>>1); i++)
		*(s+i) = *(p+i+(shift>>1)) -1/p_kesa*(*(d+i));
    /* d4 */
	for (i=0; i<(buflen>>1); i++)
		*(buffer+i+(buflen>>1)) = *(d+i) +(p_kesa-1)*(*(s+i));

	/* s4 */
	for (i=0; i<(buflen>>1); i++)
		*(buffer+i) = *(s+i) + *(buffer+i+(buflen>>1));

/*	for (i=0; i<(buflen>>1); i++){
   *(buffer+i) = p_kesa*(*(p+i+(shift>>1)));  
 	 *(buffer+i+(buflen>>1)) = 1/p_kesa*(*(p+i+(buflen>>1)+shift+(shift>>1)));	
	}
*/
    delete []p;
	delete []d;
	delete []s;
    
}

void IDwt1D(double *buffer, int buflen)
{
	int i;
    double *p1, *p2, *s, *d;
	p1 = new double [(buflen>>1)+shift];
	p2 = new double [(buflen>>1)+shift];
	s  = new double [(buflen>>1)+shift];
	d  = new double [(buflen>>1)+shift];

   /* s3 */
   for (i=0; i<(buflen>>1); i++)
	   *(s+i) = *(buffer+i) - *(buffer+i+(buflen>>1));
   /* d3 */
   for (i=0; i<(buflen>>1); i++)
	   *(d+i) =*(buffer+i+(buflen>>1))- (p_kesa-1)*(*(s+i));
   /*  s2 */
   for (i=0; i<(buflen>>1); i++)
	   *(buffer+i) = *(s+i) + (1/p_kesa)*(*(d+i));
   /*  d2 */
   for (i=0; i<(buflen>>1); i++)
	   *(buffer+i+(buflen>>1)) = *(d+i) - p_kesa*(1-p_kesa)*(*(buffer+i));


	for (i=0; i<(shift>>1); i++){
		p1[i] = buffer[i+(buflen>>1)-(shift>>1)];
		p1[i+(shift>>1)+(buflen>>1)] = buffer[i];
		p2[i] = buffer[i+buflen-(shift>>1)];
		p2[i+(shift>>1)+(buflen>>1)] = buffer[i+(buflen>>1)];
	}
	for (i=0; i<(buflen>>1); i++){
		p1[i+(shift>>1)] = buffer[i];
		p2[i+(shift>>1)] = buffer[i+(buflen>>1)];
	}


   	for (i=0; i<(shift>>1); i++){
		p1[i] = buffer[(shift>>1)-i];
		p1[i+(shift>>1)+(buflen>>1)] = buffer[(buflen>>1)-i-1];
		p2[i] = buffer[(buflen>>1)+(shift>>1)-i-1];
		p2[i+(shift>>1)+(buflen>>1)] = buffer[buflen-i-2];
	}
	for (i=0; i<(buflen>>1); i++){
		p1[i+(shift>>1)] = buffer[i];
		p2[i+(shift>>1)] = buffer[i+(buflen>>1)];
	}


   /*  s1  */
   for (i=0; i<(buflen>>1)+shift-1; i++)
	   *(s+i+1) = *(p1+i+1) - p_delta*( *(p2+i+1) + *(p2+i+1-1));
   /*  d1 */
   for (i=0; i<(buflen>>1)+shift-1; i++)
	   *(d+i) = *(p2+i) - p_gama*(*(s+i) + *(s+i+1));
    /* p1 = s0 */
   for (i=0; i<(buflen>>1)+shift-1; i++)
	   *(p1+i+1) = *(s+i+1) - p_beta*( *(d+i+1)+*(d+i+1-1));
   /*  p2 = d0 */
   for (i=0; i<(buflen>>1)+shift-1; i++)
	   *(p2+i) = *(d+i) - p_alfa*(*(p1+i) + *(p1+i+1));

   for (i=0; i<(buflen>>1); i++){
	   *(buffer+2*i) = *(p1+i+(shift>>1)) ;
	   *(buffer+2*i+1) = *(p2+i+(shift>>1));
   }
	delete []p1;
	delete []p2;
	delete []s;
	delete []d;
}


void DwtND(double buffer[], int height, int width, int lv)
{
	int i,j,k;
	int nheight,nwidth;
 
	for ( k=0; k<lv; k++){

       nheight=height>>k;
	   nwidth=width>>k;

	   double *pdata;
	   pdata = new double [nwidth];
		for (i=0; i<nheight; i++){
			for(j=0; j<nwidth; j++)
				*(pdata+j) = *(buffer+i*width+j);
			Dwt1D(pdata,nwidth);
			for(j=0; j<nwidth; j++)
				*(buffer+i*width+j) = *(pdata+j);
		}
		delete []pdata;
	
		double *p1data;
		p1data = new double [nheight];
		for(j=0; j<nwidth; j++){
			for (i=0; i<nheight; i++)
				*(p1data+i)=*(buffer+i*width+j);
			Dwt1D(p1data,nheight);
			for(i=0; i<nwidth; i++)
				*(buffer+i*width+j) = *(p1data+i);
		}
		delete []p1data;
	}

}

void IDwtND(double buffer[], int height, int width, int lv)
{

	int i,j,k;
 	int nheight,nwidth;
 	for (k=0; k<lv; k++){
 		nheight=height>>(lv-k-1);
 		nwidth=width>>(lv-k-1);
 
		double *pdata;
		pdata = new double [nheight];
 		for(j=0; j<nwidth; j++){
 			for (i=0; i<nheight; i++)
 				*(pdata+i) = *(buffer+i*width+j);
 			IDwt1D(pdata,nheight);
 			for(i=0; i<nwidth; i++)
 				*(buffer+i*width+j) = *(pdata+i);
 		}
 		delete []pdata;

		double *p1data;
		p1data = new double [nwidth];
 		for (i=0; i<nheight; i++){
 			for(j=0; j<nwidth; j++)
 				*(p1data+j)=*(buffer+i*width+j);
 			IDwt1D(p1data,nwidth);
 			for(j=0; j<nwidth; j++)
 				*(buffer+i*width+j) = *(p1data+j);
 		}
		delete []p1data;
 	}

}

void Adjust(double *ImageData,int bmpHeight,int bmpWidth, double *SparseMat,short *OutData)
{
	double fTempBufforDisp, MaxPixVal,MinPixVal,Diff;
	int x,y;
	MaxPixVal=ImageData[0];
	MinPixVal=ImageData[0];
	for( y=0; y<bmpHeight; y++)
	{
		for( x=0; x<bmpWidth; x++)
		{
			if(MaxPixVal< ImageData[y*bmpWidth+x])
				MaxPixVal=ImageData[y*bmpWidth+x];
			if(MinPixVal>ImageData[y*bmpWidth+x])
				MinPixVal=ImageData[y*bmpWidth+x];
			SparseMat[y*bmpWidth+x]=ImageData[y*bmpWidth+x];
		}
	}
	Diff=MaxPixVal-MinPixVal;
	for( y=0;y<bmpHeight/2;y++)
		 for( x=0;x<bmpWidth;x++)
		 {
			 fTempBufforDisp=ImageData[y*bmpWidth+x];
			  fTempBufforDisp-=MinPixVal;
			 fTempBufforDisp*=255;
			 fTempBufforDisp/=Diff;
			 OutData[y*bmpWidth+x]=(short) fTempBufforDisp;
		}
	for( y=bmpHeight/2;y<bmpHeight;y++)
		 for( x=0;x<bmpWidth;x++)
		 {
			 fTempBufforDisp=ImageData[y*bmpWidth+x];
			  fTempBufforDisp-=MinPixVal;
			 fTempBufforDisp*=255;
			 fTempBufforDisp/=Diff;
			OutData[y*bmpWidth+x]=(short) fTempBufforDisp;
		 }
	for( y=bmpHeight*3/4;y<bmpHeight;y++)
		 for( x=0;x<bmpWidth/2;x++)
		 {
			  fTempBufforDisp=ImageData[y*bmpWidth+x];
			   fTempBufforDisp-=MinPixVal;
			 fTempBufforDisp*=255;
			 fTempBufforDisp/=Diff;
			OutData[y*bmpWidth+x]=(short) fTempBufforDisp;
		}
	for( y=bmpHeight/2;y<bmpHeight*3/4;y++)
		 for( x=0;x<bmpWidth/2;x++)
		 {
			  fTempBufforDisp=ImageData[y*bmpWidth+x];
			   fTempBufforDisp-=MinPixVal;
			 fTempBufforDisp*=255;
			 fTempBufforDisp/=Diff;
			OutData[y*bmpWidth+x]=(short) fTempBufforDisp;
		 }

}
