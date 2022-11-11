FLAGS=-std=c++17 -Iinclude -lcurses `pkg-config --libs opencv4`

comp: main.cpp source/*.cpp
	g++ main.cpp source/*.cpp -o solosis -O3 ${FLAGS}

comp-debug: main.cpp source/*.cpp
	g++ main.cpp source/*.cpp -o solosis-debug -g ${FLAGS}