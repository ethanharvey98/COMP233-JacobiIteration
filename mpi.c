// Ethan Harvey ~ COMP 233 ~ Jacobi Iteration
// Jacobi Iteration for Steady State Heat Distribution in a 2D Plate
// Code by Argonne National Laboratory
// https://www.mcs.anl.gov/research/projects/mpi/tutorial/mpiexmpl/src/jacobi/C/main.html

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include <unistd.h>

int main(int argc, char** argv) {

	// Max number of iterations
	const int MAXITERATIONS = 500000;
	// Dimensions of the 2D array
	const int MAXN = 1000;
	// Starting north value
	const float NORTH = 100.0;
	// Starting south value
	const float SOUTH = 100.0;
	// Starting east value
	const float EAST = 0.0;
	// Starting west value
	const float WEST = 0.0;
	// Starting interior value
	const float INTERIOR = (NORTH + SOUTH + EAST + WEST) / 4.0;
	// Iteration variables
	int rank, value, size, errcnt, toterr, r, c, i, itcnt;
	// End values and number extra rows
	int i_first, i_last, remainder;
	// MPI status
	MPI_Status status;
	// diffnorm and gdiffnorm
	float diffnorm, gdiffnorm;
	// Array for old values
	float* xlocal;
	// Array for new values
	float* xnew;
	// Full array to print at the end
	float* xfull;
	// Time variables
	double start, stop, total;
	// Number of rows for each process
	int chunkrows;

	// MPI Init
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	// Start timer
	start = MPI_Wtime();

	if (rank == 0) {
		// Start header
		printf("Ethan Harvey ~ COMP 233 ~ Jacobi Iteration\n\n");
	}
	
	// Set i_first and i_last
	i_first = 1;
	i_last = MAXN / size;

	// If first or last process edit variables
	remainder = MAXN % size;
	chunkrows = MAXN / size;
	if (rank == 0) {
		i_first++;
		chunkrows += remainder;
		i_last += remainder;
	}
	if (rank == size - 1) {
		i_last--;
	}

	// xlocal
	xlocal = (float*)malloc((chunkrows+2) * MAXN * sizeof(float));

	// xnew
	xnew = (float*)malloc((chunkrows+2) * MAXN * sizeof(float));

	// xfull
	xfull = (float*)malloc(MAXN * MAXN * sizeof(float));

	/* Fill the data as specified */
	for (r = 0; r < chunkrows+2; r++) {
		for (c = 0; c < MAXN; c++) {
			*(xlocal + (r * MAXN) + c) = INTERIOR;
			*(xnew + (r * MAXN) + c) = INTERIOR;
		} // End c for loop
		// Set east and west value
		*(xlocal + (r * MAXN) + (MAXN-1)) = EAST;
		*(xlocal + (r * MAXN)) = WEST;
		*(xnew + (r * MAXN) + (MAXN-1)) = EAST;
		*(xnew + (r * MAXN)) = WEST;
	} // End r for loop
	for (c = 0; c < MAXN; c++) {
		if (rank == 0) {
			*(xlocal + ((i_first - 1) * MAXN) + c) = NORTH;
			*(xnew + ((i_first - 1) * MAXN) + c) = NORTH;
		}
		if (rank == size - 1) {
			*(xlocal + ((i_last + 1) * MAXN) + c) = SOUTH;
			*(xnew + ((i_last + 1) * MAXN) + c) = SOUTH;
		}
	} // End c for loop

	itcnt = 0;
	do {
		/* Send up unless I'm at the top, then receive from below */
		if (rank < size - 1)
			MPI_Send(xlocal + ((chunkrows) * MAXN), MAXN, MPI_FLOAT, rank + 1, 0,
				MPI_COMM_WORLD);
		if (rank > 0)
			MPI_Recv(xlocal, MAXN, MPI_FLOAT, rank - 1, 0,
				MPI_COMM_WORLD, &status);
		/* Send down unless I'm at the bottom */
		if (rank > 0)
			MPI_Send(xlocal + MAXN, MAXN, MPI_FLOAT, rank - 1, 1,
				MPI_COMM_WORLD);
		if (rank < size - 1)
			MPI_Recv(xlocal + (chunkrows + 1) * MAXN, MAXN, MPI_FLOAT, rank + 1, 1,
				MPI_COMM_WORLD, &status);

		/* Compute new values (but not on boundary) */
		itcnt++;
		diffnorm = 0.0;
		for (r = i_first; r <= i_last; r++) {
			for (c = 1; c < MAXN - 1; c++) {
				*(xnew + (r * MAXN) + c) = (*(xlocal + (r * MAXN) + (c+1)) + *(xlocal + (r * MAXN) + (c-1)) +
					*(xlocal + (r+1) * MAXN + c) + *(xlocal + (r-1) * MAXN + c)) / 4.0;
				diffnorm += (*(xnew + (r * MAXN) + c) - *(xlocal + (r * MAXN) + c)) *
					(*(xnew + (r * MAXN) + c) - *(xlocal + (r * MAXN) + c));
			} // End c for loop
		} // End r for loop

		/* Only transfer the interior points */
		float* temp = xlocal;
		xlocal = xnew;
		xnew = temp;

		MPI_Allreduce(&diffnorm, &gdiffnorm, 1, MPI_FLOAT, MPI_SUM,
			MPI_COMM_WORLD);
		
		gdiffnorm = sqrt(gdiffnorm);

		// If master print ever 1000 iterations
		if (rank == 0) {
			if (itcnt % 1000 == 0) {
				printf("At iteration %d, diff is %e\n", itcnt, gdiffnorm);
			}
		}

	} while (gdiffnorm > 1.0e-2 && itcnt < MAXITERATIONS);

	// Send all chunks back to master
	if (rank == 0) {
		// Init
		for (r = 0; r < MAXN; r++) {
			for (c = 0; c < MAXN; c++) {
				*(xfull + (r * MAXN) + c) = 0;
			} // End c for loop
		} // End r for loop
		// Add masters chunk to xfull
		for (r = 1; r < chunkrows+1; r++) {
			for (c = 0; c < MAXN; c++) {
				*(xfull + ((r-1) * MAXN) + c) = *(xnew + (r * MAXN) + c);
			} // End c for loop
		} // End r for loop
		// recieve and add other chunks
		for (i = 1; i < size; i++) {
			MPI_Recv(xfull + (MAXN * chunkrows) + (MAXN/size * MAXN * (i-1)), MAXN * (MAXN/size), MPI_FLOAT, i, 2,
				MPI_COMM_WORLD, &status);
		} // End i for loop
	}
	else {
		MPI_Send(xnew + MAXN, MAXN * (MAXN/size), MPI_FLOAT, 0, 2,
				MPI_COMM_WORLD);
	}

	// If master print to file
	if (rank == 0) {
		FILE* fp;
		fp = fopen("jacobi.ppm", "w");
		if (fp != NULL) {
			// Print header to file
			fprintf(fp, "P3\n%d %d\n", MAXN, MAXN);
			fprintf(fp, "# Ethan Harvey ~ COMP 233 ~ Laplace Heat Distribution\n");
			fprintf(fp, "# This image took %d iterations to converge\n255\n", itcnt);
			for (r = 0; r < MAXN; r++) {
				for (c = 0; c < MAXN; c++) {
					// Calculate each pixel value
					fprintf(fp, "%d 0 %d ", (int)(*(xfull + (r * MAXN) + c) / 100 * 255), (int)(255 - (*(xfull + (r * MAXN) + c) / 100 * 255)));
					if ((c + 1) % 5 == 0) {
						fprintf(fp, "\n");
					}
				}
			}
			fclose(fp);
		}
	} // End print

	// End timer
	stop = MPI_Wtime();
	total = stop - start;

	MPI_Finalize();

	// Clean up
	free(xlocal);
	free(xnew);
	free(xfull);

	if (rank == 0) {
		// Normal Termination
		printf("Time: %f\n", total);
		printf("\n\n<<< Normal Termination >>>\n\n");
	}
	return 0;
}