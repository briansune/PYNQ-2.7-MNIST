template <
	int InCH,
	int InDim,
	int Poolsize>
void Pool(
	hls::stream<AXI_VAL> &stream_in,
	hls::stream<AXI_VAL> &stream_out)
{

	static NET_VAL A[InDim][InDim][InCH];

	AXI_VAL Inbuf, Outbuf;

#pragma HLS BIND_STORAGE variable=A type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=A dim=2 factor=InDim cyclic

	// first data showing mode.
	// 0 - CNN forward propagation
	// 1 - weight loading
	Inbuf = stream_in.read();
	AXI_VAL status = Inbuf;
	stream_out.write(Inbuf);

	Inbuf = stream_in.read();
	AXI_VAL batch_size = Inbuf;
	stream_out.write(Inbuf);

	Inbuf = stream_in.read();
	AXI_VAL KernelDim = Inbuf;
	stream_out.write(Inbuf);

	Inbuf = stream_in.read();
	AXI_VAL IFMCH = Inbuf;
	stream_out.write(Inbuf);

	Inbuf = stream_in.read();
	AXI_VAL IFMDim = Inbuf;
	stream_out.write(Inbuf);

	Inbuf = stream_in.read();
	AXI_VAL OFMCH = Inbuf;
	stream_out.write(Inbuf);

	Inbuf = stream_in.read();
	AXI_VAL OFMDim = Inbuf;
	stream_out.write(Inbuf);

	if (status == 0) // execution
	{
		for (int num_img = 0; num_img < batch_size; num_img++)
		{
			// ==================================================================
#pragma HLS LOOP_TRIPCOUNT avg=5000 max=10000 min=1
			for (int j = 0; j < InDim; ++j){
				for (int k = 0; k < InDim; ++k){
					for (int i = 0; i < InCH; ++i)
					{
#pragma HLS PIPELINE II=1
						Inbuf = stream_in.read();
						A[j][k][i] = Inbuf;
					}
				}
			}
			// ==================================================================

			// ==================================================================
			NET_VAL buf;

			for (int ia = 0; ia < InDim/Poolsize; ia++){
				for (int ib = 0; ib < InDim/Poolsize; ib++){
					for (int i = 0; i < InCH; i++){
						buf = A[ia*Poolsize][ib*Poolsize][i];

						for (int ka=0; ka < Poolsize; ka++){
							for (int kb=0; kb < Poolsize; kb++){
#pragma HLS PIPELINE II=1
								buf = MAX2(buf, A[ia*Poolsize + ka][ib*Poolsize + kb][i]);
							}
						}
						Outbuf = buf;
						stream_out.write(Outbuf);
					}
				}
			}
		}
	}
	else // pass filters for other layers
	{
		int KER_bound = OFMCH * IFMCH * KernelDim * KernelDim;
		for (int i = 0; i < KER_bound; i++)
		{
#pragma HLS LOOP_TRIPCOUNT avg=38372 max=147456 min=144
#pragma HLS PIPELINE II=1
			Inbuf = stream_in.read();
			stream_out.write(Inbuf);
		}
	}
}
