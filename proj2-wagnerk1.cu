/* ==================================================================
	Programmers:
	Kevin Wagner
	Elijah Malaby
	John Casey

	Omptimizing SDH histograms for input larger then global memory
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>


#define BOX_SIZE 23000 /* size of the data box on one dimension */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	float x_pos;
	float y_pos;
	float z_pos;
} atom;

unsigned long long * histogram;		/* list of all buckets in the histogram */
unsigned long long  PDH_acnt;	/* total number of data points */
int block_size;		/* Number of threads per block */
int num_buckets;	/* total number of buckets in the histogram */
float   PDH_res;	/* value of w */
atom * atom_list;	/* list of all data points */
unsigned long long * histogram_GPU;
unsigned long long * temp_histogram_GPU;
atom * atom_list_GPU;

__global__ void kernelSumHistogram( unsigned long long int *InputHists, unsigned long long int *hist, int num_atoms, int num_buckets, int block_size) {
  unsigned long long int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int h_pos = tid;
  unsigned long long int NumberOfSumLoop = 0;
  NumberOfSumLoop = (num_atoms)/block_size + ((num_atoms%block_size) ? 1:0);

  while(h_pos < num_buckets) {
    unsigned long long int tmpAns = 0;
    for(int i=0;i<NumberOfSumLoop;i++){
      tmpAns = tmpAns + *(InputHists+(i*num_buckets)+h_pos);
    }
    hist[h_pos] = tmpAns;
    h_pos += blockDim.x * gridDim.x;
  }
  __syncthreads();
}

__device__ void block_to_block (atom * block_a, atom * block_b, int b_length, bucket * histogram, float resolution) {
  atom me = block_a[threadIdx.x];
  for(int i = 0; i < b_length; i++)
    atomicAdd(&(histogram[(int)(sqrt((me.x_pos - block_b[i].x_pos) * (me.x_pos - block_b[i].x_pos) +
                                     (me.y_pos - block_b[i].y_pos) * (me.y_pos - block_b[i].y_pos) +
                                     (me.z_pos - block_b[i].z_pos) * (me.z_pos - block_b[i].z_pos)) / resolution)]),
              1);
}

 __global__ void GPUKernelFunction (unsigned long long PDH_acnt, float PDH_res, atom * atom_list_GPU, unsigned long long * histogram_GPU, int num_buckets) {

  extern __shared__ unsigned long long SHist[];
	/* assign register values */
	int i, h_pos;
	float dist;
  atom * my_block = &atom_list_GPU[blockIdx.x * blockDim.x];
  atom temp_atom_1 = my_block[threadIdx.x];

  for(h_pos=threadIdx.x; h_pos < num_buckets; h_pos+=blockDim.x)
    SHist[h_pos] = 0;

  __syncthreads();

	/* loop through all points in atom list calculating distance from current point to all further points */
	for(i = threadIdx.x + 1; i < blockDim.x; i++)
	{
		atom temp_atom_2 = my_block[i];
		dist = sqrt((temp_atom_1.x_pos - temp_atom_2.x_pos) * (temp_atom_1.x_pos - temp_atom_2.x_pos) +
                (temp_atom_1.y_pos - temp_atom_2.y_pos) * (temp_atom_1.y_pos - temp_atom_2.y_pos) +
                (temp_atom_1.z_pos - temp_atom_2.z_pos) * (temp_atom_1.z_pos - temp_atom_2.z_pos));
		h_pos = (int)(dist / PDH_res);
		atomicAdd(&(SHist[h_pos]), 1);
	}
  __syncthreads();
  for(i=blockIdx.x+1; i < gridDim.x-1; i++)
    block_to_block(my_block,
                   &atom_list_GPU[i*blockDim.x],
                   blockDim.x,
                   SHist,
                   PDH_res);
  block_to_block(my_block,
                 &atom_list_GPU[i*blockDim.x],
                 PDH_acnt-i*blockDim.x, // Last block may be small
                 SHist,
                 PDH_res);
  __syncthreads();
  for(h_pos = threadIdx.x; h_pos < num_buckets; h_pos += blockDim.x)
    *(histogram_GPU+(num_buckets*blockIdx.x)+h_pos) += SHist[h_pos];
}

/* print the counts in all buckets of the histogram  */
void output_histogram_GPU(){
	int i;
	unsigned long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i]);
		total_cnt += histogram[i];
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

void GPU_baseline() {

  int num_blocks = ((PDH_acnt + block_size)/block_size);
	/* copy histogram to device memory */
	cudaMalloc((void**) &histogram_GPU, sizeof(unsigned long long)*num_buckets);
	cudaMemset(histogram_GPU, 0, sizeof(unsigned long long)*num_buckets);
	cudaMalloc((void**) &temp_histogram_GPU, sizeof(unsigned long long)*num_buckets*num_blocks);
	cudaMemset(temp_histogram_GPU, 0, sizeof(unsigned long long)*num_buckets*num_blocks);


	/* copy atom list to device memory */
	cudaMalloc((void**) &atom_list_GPU, sizeof(atom) * PDH_acnt);
	cudaMemcpy(atom_list_GPU, atom_list, sizeof(atom) * PDH_acnt, cudaMemcpyHostToDevice);

	/* start time keeping */
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );

	/* Run Kernel */
	GPUKernelFunction <<<num_blocks, block_size, sizeof(unsigned long long)*num_buckets>>> (PDH_acnt, PDH_res, atom_list_GPU, temp_histogram_GPU, num_buckets);
  cudaDeviceSynchronize();
  kernelSumHistogram<<<3, 512>>>(temp_histogram_GPU, histogram_GPU, PDH_acnt, num_buckets, block_size);

	/* stop time keeping */
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );

	/* transfer histogram to host memory */
	cudaMemcpy(histogram, histogram_GPU, sizeof(unsigned long long)*num_buckets, cudaMemcpyDeviceToHost);

	/* print out the histogram */
	output_histogram_GPU();
	elapsedTime = elapsedTime/1000;
	printf( "******** Total Running Time of Kernel = %0.5f sec *******\n", elapsedTime );

	/* free cuda timekeeping */
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
  cudaFree(temp_histogram_GPU);
}

/* Input Validation Function */
bool isNumber(char number[], bool floatingpoint)
{
    for (int i = 0; number[i] != 0; i++)
    {
        //if (number[i] > '9' || number[i] < '0')
        if (!isdigit(number[i]))
		{
			if((number[i] == '.' && floatingpoint))
			{
				floatingpoint = false;
			}
			else
			{
				return false;
			}
		}
    }
    return true;
}


/* Most of this input validation can probably be pulled whenever we hardcode our block size and if we hardcode our bucket width */
int main(int argc, char **argv)
{
	/* input validation */
	if((argc > 3))
	{
		if(((isNumber(argv[1], false) && isNumber(argv[2], true)) && isNumber(argv[3], false)))
		{
			PDH_acnt = atoi(argv[1]);
			PDH_res	 = atof(argv[2]);
			block_size = atoi(argv[3]);
		}
		else
		{
			printf( "Invalid Input Error Invalid Arguments\n Valid input is ./program_name {#of_samples} {bucket_width} {block_size}\n");
			return 0;
		}
	}
	else
	{
		printf( "Invalid Input Error Insufficient Arguments\n Valid input is ./program_name {#of_samples} {bucket_width} {block_size}\n");
		return 0;
	}

	/* allocate memory */
	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (unsigned long long *)malloc(sizeof(unsigned long long)*num_buckets);
	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	srand(1);
	/* generate data following a uniform distribution */
	for(int i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((float)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((float)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((float)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	/* call GPU histrogram compute */
	GPU_baseline();

	/* free memory */
	free(histogram);
	free(atom_list);

	return 0;
}


