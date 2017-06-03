rwildcard=$(wildcard $1$2) $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2))

CC=g++
SRC=$(call rwildcard,./,*.cpp)
HEADERS=$(call rwildcard,./,*.hpp)
OBJECTS=$(SRC:.cpp=.o)
EXECUTABLE=cnn
CFLAGS=-c -Wall -std=c++11 -I/opt/AMDAPP/include
LDFLAGS=-L/opt/AMDAPP/lib/x86_64/ -lOpenCL

all : $(SRC) $(HEADERS) $(EXECUTABLE)
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS) -std=c++11
.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean :
	rm -rf $(OBJECTS) $(EXECUTABLE)
remake :
	make clean ; make