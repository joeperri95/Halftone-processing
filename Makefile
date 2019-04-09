CC = g++

CFLAGS = -Wall -std=c++11

SRC = main
DEPS = main.hpp
OBJECTS = $(SRC:.cpp:=.o)

EXECUTABLE = run

OPENCV = $(shell pkg-config --cflags --libs opencv)
LIBS = -lm -lpthread 

all: $(OBJECTS)
	$(CC) $(OBJECTS).o -o $(EXECUTABLE) $(CFLAGS) $(OPENCV) $(LIBS)

$(OBJECTS):
	$(CC) $(SRC).cpp $(CFLAGS) $(DEPS) -c

.PHONY: clean

clean: 
	rm -f $(EXECUTABLE) $(wildcard *.o) $(wildcard out*.png)