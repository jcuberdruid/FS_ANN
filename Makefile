CXX = clang++
CXXFLAGS = -Ofast -std=c++20
TARGET = runmain.out
SRC = main.cpp 

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

.PHONY: clean
clean:
	rm -f $(TARGET)

