LINK :=g++ -fPIC
CXX :=g++
CS: data_operate.o DWT.o CompressedSample.o Cosamp.o main.o  cosamp_opencl.o 
	g++  data_operate.o DWT.o  CompressedSample.o Cosamp.o main.o  cosamp_opencl.o -lgdal -L/usr/local/lib -lOpenCL -L/usr/local/clBLAS-2.2.0-Linux-x64/lib64 -lclBLAS -o CS
data_operate.o:data_operate.cpp
	$(CXX) -c data_operate.cpp -lgdal -o data_operate.o
DWT.o:DWT.cpp
	g++ -c DWT.cpp -o DWT.o
CompressedSample.o:CompressedSample.cpp
	$(CXX) -c CompressedSample.cpp -o CompressedSample.o
Cosamp.o:Cosamp.cpp
	$(CXX) -c Cosamp.cpp -o Cosamp.o
main.o:main.cpp
	$(CXX) -c main.cpp -o main.o
cosamp_opencl.o:cosamp_opencl.cpp
	$(CXX) -c cosamp_opencl.cpp -o cosamp_opencl.o
clean:
	rm CS *.o	
