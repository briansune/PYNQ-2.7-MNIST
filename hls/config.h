#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>

#ifndef AXI_VAL_DEF

typedef ap_int<32> AXI_CAL;
typedef ap_int<16> AXI_VAL;
typedef ap_int<8> NET_VAL;

typedef qdma_axis<16,0,0,0> AXI_DMA_IF;

#define AXI_VAL_DEF
#endif

AXI_CAL MAX(AXI_CAL x, AXI_CAL y){
	if (x > y)
		return x;
	else
		return y;
}

NET_VAL MAX2(NET_VAL x, NET_VAL y){
	if (x > y)
		return x;
	else
		return y;
}
