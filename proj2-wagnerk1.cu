/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	GPU algorithm implementation for 3D data
	To compile: nvcc proj2-wagnerk1-naive-solution.cu -o GPU_V1 in the rc machines
	
	Additional Programming By Kevin Wagner U75723519
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>


#define BOX_SIZE 23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a unsigned long long type as the count might be huge */
} bucket;

bucket * histogram;		/* list of all buckets in the histogram   */
unsigned long long  PDH_acnt;	/* total number of data points            */
int block_size;			/* Number of threads per block */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */
bucket * histogram_GPU;	
atom * atom_list_GPU;
	


__global__ void GPUKernelFunction (unsigned long long PDH_acnt, double PDH_res, atom * atom_list_GPU, bucket * histogram_GPU) {
	
	int threadID = threadIdx.x + blockIdx.x * blockDim.x;
	int i, h_pos;
	double dist;
	atom temp_atom_1 = atom_list_GPU[threadID];
	
	for(i = threadID + 1; i < PDH_acnt; i++)
	{
		atom temp_atom_2 = atom_list_GPU[i];
		dist = sqrt((temp_atom_1.x_pos - temp_atom_2.x_pos) * (temp_atom_1.x_pos - temp_atom_2.x_pos) + 
					(temp_atom_1.y_pos - temp_atom_2.y_pos) * (temp_atom_1.y_pos - temp_atom_2.y_pos) + 
					(temp_atom_1.z_pos - temp_atom_2.z_pos) * (temp_atom_1.z_pos - temp_atom_2.z_pos));
			
		h_pos = (int)(dist / PDH_res);
		atomicAdd(&(histogram_GPU[h_pos].d_cnt), 1);
	}
}

/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram_GPU(){
	int i; 
	unsigned long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}

void GPU_baseline() {
	

	

	/* allocate GPU histogram */
	cudaMalloc((void**) &histogram_GPU, sizeof(bucket)*num_buckets);
	cudaMemcpy(histogram_GPU, histogram, sizeof(bucket)*num_buckets, cudaMemcpyHostToDevice);
	
	/* copy atom list to device memory */
	cudaMalloc((void**) &atom_list_GPU, sizeof(atom) * PDH_acnt);
	cudaMemcpy(atom_list_GPU, atom_list, sizeof(atom) * PDH_acnt, cudaMemcpyHostToDevice);
	
	output_histogram_GPU();
	
	/* start time keeping */
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );
	
	/* Run Kernel */
	GPUKernelFunction <<<(ceil(PDH_acnt + block_size)/block_size), block_size>>> (PDH_acnt, PDH_res, atom_list_GPU, histogram_GPU);
	
	/* stop time keeping */
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );
	
	/* transfer histogram to host memory */
	cudaMemcpy(histogram, histogram_GPU, sizeof(bucket)*num_buckets, cudaMemcpyDeviceToHost);
	
	/* print out the histogram */
	output_histogram_GPU();
	elapsedTime = elapsedTime/1000;
	printf( "******** Total Running Time of Kernel = %0.5f sec *******\n", elapsedTime );
	
	/* free cuda timekeeping */
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
}

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

	
	int i;
	
	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);
	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	
	/* call GPU histrogram compute */
	GPU_baseline();

	/* free memory */
	free(histogram);
	free(atom_list);
	
	return 0;
}


