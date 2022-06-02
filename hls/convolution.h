template <
    int InCH,
    int InDim,
    int OutCH,
    int KerDim>
void Conv(
    hls::stream<AXI_VAL> &stream_in,
    hls::stream<AXI_VAL> &stream_out,
    const int layer_id,
    const int output_rectify)
{
	// Raw dat of the previous layer
    static NET_VAL A[InCH][InDim][InDim];

    // Weight of the convolution filters
    static NET_VAL B[OutCH][InCH][KerDim][KerDim];

#pragma HLS BIND_STORAGE variable=A type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=B type=ram_2p impl=lutram

#pragma HLS ARRAY_PARTITION variable=A block factor=InDim dim=3
#pragma HLS ARRAY_PARTITION variable=B block factor=KerDim dim=4

    AXI_VAL tmp_val;

    // first data showing mode.
    // 0 - CNN forward propagation
    // 1 - weight loading
    tmp_val = stream_in.read();
    AXI_VAL status = tmp_val;
    stream_out.write(tmp_val);

    tmp_val = stream_in.read();
    AXI_VAL batch_size = tmp_val;
    stream_out.write(tmp_val);

    tmp_val = stream_in.read();
    AXI_VAL KernelDim = tmp_val;
    stream_out.write(tmp_val);

    tmp_val = stream_in.read();
    AXI_VAL IFMCH = tmp_val;
    stream_out.write(tmp_val);

    tmp_val = stream_in.read();
    AXI_VAL IFMDim = tmp_val;
    stream_out.write(tmp_val);

    tmp_val = stream_in.read();
    AXI_VAL OFMCH = tmp_val;
    stream_out.write(tmp_val);

    tmp_val = stream_in.read();
    AXI_VAL OFMDim = tmp_val;
    stream_out.write(tmp_val);

    if (status == layer_id) // store weight for current layer
    {
    	weight_load:
		for (int i = 0; i < OutCH; i++)
            for (int j = 0; j < InCH; j++)
                for (int ka = 0; ka < KerDim; ka++)
                    for (int kb = 0; kb < KerDim; kb++)
                    {
#pragma HLS PIPELINE II=1
                        tmp_val = stream_in.read();
                        B[i][j][ka][kb] = ap_int<8>(tmp_val);
                        stream_out.write(B[i][j][ka][kb]);
                    }
    }
    else if (status == 0) // execute
    {
    	inference_top:
		for (int num_img = 0; num_img < batch_size; num_img++)
        {
			// ============================================================================
#pragma HLS LOOP_TRIPCOUNT avg=64 max=127 min=1
    		inference_a_ld0:
			for (int j = 0; j < InDim; j++){
				inference_a_ld1:
                for (int k = 0; k < InDim; k++)
                {
					inference_a_ld2:
                    for (int i = 0; i < InCH; i++)
                    {
#pragma HLS PIPELINE II=1
                        tmp_val = stream_in.read();
                        A[i][j][k] = tmp_val;
                    }
                }
			}
			// ============================================================================
			AXI_VAL relu_cal;

			inference_cal_l0:
            for (int ia = 0; ia < InDim - (KerDim - 1); ia++)
            {
				inference_cal_l1:
                for (int ib = 0; ib < InDim - (KerDim - 1); ib++)
                {
					inference_cal_l2:
                    for (int i = 0; i < OutCH; i++)
                    {
                    	AXI_VAL buf = 0;
                        inference_cal_l3:
						for (int j = 0; j < InCH; j++)
                        {
#pragma HLS PIPELINE II=1
							inference_cal_l4:
                            for (int ka = 0; ka < KerDim; ka++){
								inference_cal_l5:
                                for (int kb = 0; kb < KerDim; kb++){
                                    buf += (A[j][ia + ka][ib + kb] * B[i][j][ka][kb]) >> quant_scale;
                                }
                            }
                        }
                        relu_cal = (output_rectify) ? (MAX(0, buf)) : (buf);
                        stream_out.write(relu_cal);
                    }
                }
            }
        }
    }
    else // pass filters for other layers
    {
        int KER_bound = OFMCH * IFMCH * KernelDim * KernelDim;

        bypass_top:
        for (int i = 0; i < KER_bound; i++)
        {
#pragma HLS LOOP_TRIPCOUNT avg=38372 max=147456 min=144
#pragma HLS PIPELINE II=1
            tmp_val = stream_in.read();
            stream_out.write(tmp_val);
        }
    }
}
