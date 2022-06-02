template <
	int InCH,
	int InDim,
	int Poolsize>
void Pool(
	hls::stream<AXI_VAL> &stream_in,
	hls::stream<AXI_VAL> &stream_out,
	const int Poolmode) // 0 for max pooling, 1 for average pooling
{

	static NET_VAL A[InCH][InDim][InDim];

	AXI_VAL Inbuf, Outbuf;

#pragma HLS BIND_STORAGE variable=A type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=A dim=3 factor=InDim block

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
#pragma HLS LOOP_TRIPCOUNT avg=64 max=127 min=1
			for (int j = 0; j < InDim; ++j){
				for (int k = 0; k < InDim; ++k){
					for (int i = 0; i < InCH; ++i)
					{
#pragma HLS PIPELINE II=1
						Inbuf = stream_in.read();
						A[i][j][k] = Inbuf;
					}
				}
			}
			// ==================================================================

			// ==================================================================
			for (int ia = 0; ia < InDim; ia += Poolsize)
			{
				for (int ib = 0; ib < InDim; ib += Poolsize)
				{
					for (int i = 0; i < InCH; i++)
					{
						NET_VAL buf;
						if (Poolmode == 1)
						{
							buf = A[i][ia][ib];
							for (int k=0; k < Poolsize * Poolsize; k++)
							{
#pragma HLS PIPELINE II=1
								buf += A[i][ia + (k / Poolsize)][ib + (k % Poolsize)];
							}
							buf /= Poolsize * Poolsize;
						}
						else
						{
							buf = A[i][ia][ib];
							for (int k=0; k < Poolsize * Poolsize; k++)
							{
#pragma HLS PIPELINE II=1
								buf = MAX(buf, A[i][ia + (k / Poolsize)][ib + (k % Poolsize)]);
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
