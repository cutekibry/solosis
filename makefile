comp:
	g++ *.cpp -Iinclude -o main -g

compfast:
	g++ *.cpp -Iinclude -o main -O3 -ffast-math

run:
	g++ *.cpp -Iinclude -o main -g
	./main

runfast:
	g++ *.cpp -Iinclude -o main -O3 -ffast-math
	./main