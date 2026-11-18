CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra
cagran: src/cagran.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $< -lm
clean:
	rm -f cagran
