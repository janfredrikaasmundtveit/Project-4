# Comment lines
# General makefile for c++ - choose PROG =   name of given program
# Here we define compiler option, libraries and the  target
CPPflags= mpic++ -O3
# Here we define the library functions we nee
LIB = -larmadillo -llapack -lblas
# Here we define the name of the executable
PROG= ising.exe
${PROG} :	   	main.o 
			${CPPflags} main.o ${LIB} -o ${PROG}

	
main.o :			main.cpp 
		        	${CPPflags} -c main.cpp
	