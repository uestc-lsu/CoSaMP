#include "gdal_priv.h"
#include <iostream>
#include <iomanip> 
#include <string.h>
#include <fstream> 
#include <ctype.h>
#include "cpl_string.h"
#include <iostream>
using namespace std;
#pragma comment(lib, "gdal_i.lib")

bool readImageGDAL(unsigned char **pImageData,int &width,int &height,int &nChannels, const char *filePath,double trans[6],char pszSrcWKT[])
{
	GDALAllRegister();
	GDALDataset *poDataset = NULL;
	poDataset=(GDALDataset*) GDALOpen(filePath,GA_ReadOnly);	
	if(poDataset == NULL)
	{
	    GDALClose(poDataset);
	    return false;
	}
	width = poDataset->GetRasterXSize();
	height = poDataset->GetRasterYSize();
	
	char* hiahia; 
	hiahia = (char *)(poDataset->GetProjectionRef()); 
	int touying;
	touying=strlen(hiahia);

	strcpy(pszSrcWKT, hiahia);
	
	CPLErr aaa=poDataset->GetGeoTransform(trans);
	GDALRasterBand* pBand;
	int i = 0;
	int nRastercount = poDataset->GetRasterCount();

	if (nRastercount == 1) 
	{
	    nChannels = 1;
	    pBand = poDataset->GetRasterBand(1); 
	    *pImageData = new unsigned char[width * height];
	    pBand->RasterIO(GF_Read,
	        0,0,		
		width,height,	
		*pImageData,	
		width,height,	
		GDT_Byte,	
		0,
		0);		
		GDALClose(poDataset);
		return true;
	}
	else if ( nRastercount == 3 && (nChannels == 3 || nChannels < 0) ) 
	{		
	    nChannels = 3;
	    *pImageData = new unsigned char[nRastercount * width * height];		
	    for (i = 1; i <= nRastercount; ++ i)
	    {
		unsigned char *pImageOffset = *pImageData + i - 1;
		GDALRasterBand* pBand = poDataset->GetRasterBand(nRastercount-i+1);
			
		pBand->RasterIO(
			GF_Read,
			0,0,			
			width,height,	
			pImageOffset,	
			width,height,	
			GDT_Byte,		
			3,				
			0);				
	    }
	    GDALClose(poDataset);
	    return true;
	}
	else if ( nRastercount == 3 && nChannels == 1 ) 
	{
	    unsigned char **img = new unsigned char*[nRastercount];
	    for (i = 0; i < nRastercount; i++)
	    {
	        img[i] = new unsigned char[width * height];
	    }
	    for (i = 1; i <= nRastercount; ++ i)
	    {
	        pBand = poDataset->GetRasterBand(nRastercount-i+1); 
		pBand->RasterIO(GF_Read,
			0,0,
			width,height,
			img[i-1],
			width,height,
			GDT_Byte,
			0,
			0);
	    }
	    GDALClose(poDataset);
	    *pImageData = new unsigned char[width*height];
	    for (int r = 0; r < height; ++ r)
	    {
		for (int c = 0; c < width; ++ c)
		{
		    int t = (r*width+c);
		    (*pImageData)[t] = (img[2][t]*3 + img[1][t]*6 + img[0][t] + 5)/10;
		}
	    }
	
            for (i = 0; i < nRastercount; ++ i)
	    {
	        delete [] img[i];
	    }
	    delete []img; img = NULL;
	    return true;
	}
	else 
	{
            return false;
	}
}

char *strlwr(char *s)
{
    char *p;
    for(p=s;*p!='\0';p++)
    {
        if('A'<=(*p)&&(*p)<='Z')
            (*p)+=32;
    }
    return s;
}


char* findImageTypeGDAL( char *pDstImgFileName)
{
	char *dstExtension = strlwr(strrchr(pDstImgFileName,'.') + 1);
	char *Gtype = NULL;
	if		(0 == strcmp(dstExtension,"bmp")) Gtype =(char *)"BMP";
	else if (0 == strcmp(dstExtension,"jpg")) Gtype = (char *)"JPEG";
	else if (0 == strcmp(dstExtension,"png")) Gtype = (char *)"PNG";
	else if (0 == strcmp(dstExtension,"tif")) Gtype =(char *)"GTiff";
	else if (0 == strcmp(dstExtension,"gif")) Gtype =(char *)"GIF";
	else Gtype = NULL;

	return Gtype;
}

bool WriteImageGDAL( char* pDstImgFileName,unsigned char * pImageData,int width,int height,int nChannels,double trans[6],char *pszSrcWKT)
{

	GDALAllRegister();
	char *GType = NULL;
	GType = findImageTypeGDAL(pDstImgFileName);
	if (GType == NULL)	{ return false; }

	GDALDriver *pMemDriver = NULL;
	pMemDriver = GetGDALDriverManager()->GetDriverByName("MEM");
	if( pMemDriver == NULL ) { return false; }

	GDALDataset * pMemDataSet = pMemDriver->Create("",width,height,nChannels,GDT_Byte,NULL);
	GDALRasterBand *pBand = NULL;
	int nLineCount = width * nChannels;
	unsigned char *ptr1 = (unsigned char *)pImageData;
	for (int i = 1; i <= nChannels; i++)
	{
	    pBand = pMemDataSet->GetRasterBand(nChannels-i+1);
	    pBand->RasterIO(GF_Write, 
		    0, 
		    0, 
		    width, 
		    height, 
		    ptr1+i-1 , 
		    width, 
		    height, 
		    GDT_Byte, 
		    nChannels, 
		    nLineCount); 
	}

	GDALDriver *pDstDriver = NULL;
	pDstDriver = (GDALDriver *)GDALGetDriverByName(GType);
	if (pDstDriver == NULL) { return false; }

	pMemDataSet->SetProjection(pszSrcWKT);
	pMemDataSet->SetGeoTransform( trans );

        GDALDataset *poDstDS;
	 poDstDS = pDstDriver->CreateCopy(pDstImgFileName,pMemDataSet,FALSE, NULL, NULL, NULL);
	if( poDstDS != NULL )
		delete poDstDS;
	GDALClose(pMemDataSet); 

	return true; 
}


