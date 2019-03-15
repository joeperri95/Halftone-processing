CC = g++

CFLAGS = -Wall -std=c++11

SRC = main
DEPS = define.h
OBJECTS = $(SRC:.cpp:=.o)

EXECUTABLE = run

OPENCV = $(shell pkg-config --cflags --libs opencv)
SDL = $(shell pkg-config --cflags --libs sdl2)
ALSA = $(shell pkg-config --cflags --libs alsa)
LIBS = -lm -lpthread 

all: $(OBJECTS)
	$(CC) $(OBJECTS).o -o $(EXECUTABLE) $(CFLAGS) $(SDL) $(PNG) $(ZLIB) $(OPENCV) $(LIBS)

$(OBJECTS):
	$(CC) $(SRC).cpp $(CFLAGS) $(DEPS) -c

.PHONY: clean

clean: 
	rm -f $(EXECUTABLE) $(wildcard *.o) $(wildcard out*.png)