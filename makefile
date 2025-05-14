# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall

# Target executable
TARGET = neuralnetwork

# Source files
SOURCES = Module.cpp LinearLayer.cpp ReLUActivation.cpp Block.cpp NeuralNetwork.cpp driver-3.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Build the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(TARGET) $(OBJECTS)