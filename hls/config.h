#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>

#define quant_scale 6

#ifndef AXI_VAL_DEF

typedef ap_int<32> AXI_CAL;
typedef ap_int<16> AXI_VAL;
typedef ap_int<8> NET_VAL;

typedef qdma_axis<16,0,0,0> AXI_DMA_IF;

#define AXI_VAL_DEF
#endif


AXI_VAL MAX(AXI_VAL x, AXI_VAL y){
	if (x > y)
		return x;
	else
		return y;
}

AXI_CAL MAX2(AXI_CAL x, AXI_CAL y){
	if (x > y)
		return x;
	else
		return y;
}


