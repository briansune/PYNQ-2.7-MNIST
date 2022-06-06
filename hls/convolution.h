template <
    int InCH,
    int InDim,
    int OutCH,
    int KerDim>
void Conv(
    hls::stream<AXI_VAL> &stream_in,
    hls::stream<AXI_VAL> &stream_out,
    const int layer_id,
    const int output_rectify,
	const int reduce)
{
	// Raw dat of the previous layer
    static NET_VAL A[InDim][InDim][InCH];

    // Weight of the convolution filters
    static NET_VAL B[OutCH][KerDim][KerDim][InCH];

#pragma HLS BIND_STORAGE variable=A type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=B type=ram_2p impl=lutram

#pragma HLS ARRAY_PARTITION variable=A dim=2 complete
#pragma HLS ARRAY_PARTITION variable=B dim=3 complete

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
            for (int ka = 0; ka < KerDim; ka++)
            	for (int kb = 0; kb < KerDim; kb++)
            		for (int j = 0; j < InCH; j++){
#pragma HLS PIPELINE II=1
                        tmp_val = stream_in.read();
                        B[i][ka][kb][j] = ap_int<8>(tmp_val);
                        stream_out.write(B[i][ka][kb][j]);
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
                        A[j][k][i] = tmp_val;
                    }
                }
			}
			// ============================================================================

			inference_kin_h:
			for (int ia = 0; ia < InDim - (KerDim - 1); ia++)
			{
				inference_kin_w:
				for (int ib = 0; ib < InDim - (KerDim - 1); ib++)
				{
					inference_out_ch:
					for (int i = 0; i < OutCH; i++)
					{
						AXI_CAL buf = 0;

						inference_ftr_h:
						for (int ka = 0; ka < KerDim; ka++){
#pragma HLS PIPELINE II=1
							inference_ftr_w:
							for (int kb = 0; kb < KerDim; kb++){
								inference_in_ch:
								for (int j = 0; j < InCH; j++)
								{
                                	buf += A[ia + ka][ib + kb][j] * B[i][ka][kb][j];
                                }
                            }
                        }

						buf >>= reduce;
						buf = (output_rectify) ? MAX(0, buf) : buf;
						stream_out.write(buf);
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
