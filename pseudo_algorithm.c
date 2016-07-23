chunk chunks[];
histograms interchunk_histograms;
histograms intrachunk_histograms;
typedef struct {
  chunk c;
  event e;
} chunk_with_event
chunk_with_event *primary;
chunk_with_event *secondary;
chunk_with_event *free_chunk;
histogram final_histogram;

cudaMemcpy(primary->c, chunks[0], chunkSize, cudaMemcpyHostToDevice);
cudaEventRecord(primary->e);
for (int i = 0; i < num_chunks; i++) {
  cudaEventSynchronize(primary->e);
  intra_chunk_kernel(primary->c, intrachunk_histogram);
  if (i+1 < num_chunks) {
    cudaMemcpy(secondary->c, chunks[i+1], chunkSize, cudaMemcpyHostToDevice);
    cudaEventRecord(secondary->e);
    for (int j=i+2; j < num_chunks; j++) {
      cudaEventSynchronize(secondary->e);
      cudaMemcpy(free_chunk->c, chunks[j], chunkSize, cudaMemcpyHostToDevice);
      cudaEventRecord(free_chunk->e);
      inter_chunk_kernel(secondary->c, interchunk_histogram);
      swap(secondary, free_chunk);
    }
  }
  swap(primary, secondary);
}
sum_histogram_kernel(interchunk_histogram, final_histogram);
sum_histogram_kernel(intrachunk_histogram, final_histogram);