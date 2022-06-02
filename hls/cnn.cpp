#include "cnn.h"

void cnn(
	hls::stream<AXI_DMA_IF> &stream_in,
	hls::stream<AXI_DMA_IF> &stream_out
){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=stream_in
#pragma HLS INTERFACE axis port=stream_out
#pragma HLS DATAFLOW

	hls::stream<AXI_VAL> connect_0, connect_1, connect_2;

#pragma HLS STREAM variable=connect_0 depth=1000
#pragma HLS STREAM variable=connect_1 depth=200
#pragma HLS STREAM variable=connect_2 depth=100

	AXI_DMA_SLAVE(stream_in, connect_0);

	FC<784, 128, 8>(connect_0, connect_1, 1, 1, 8);
	
	FC<128, 10, 10>(connect_1, connect_2, 2, 1, 0);

	AXI_DMA_MASTER(connect_2, stream_out);
}
