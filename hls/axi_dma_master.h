void AXI_DMA_MASTER(
	hls::stream<AXI_VAL> &stream_in,
	hls::stream<AXI_DMA_IF> &stream_out)
{
	AXI_VAL tmp_val;
	AXI_DMA_IF Outbuf;

	tmp_val = stream_in.read();
	AXI_VAL status = tmp_val;
	Outbuf.data = tmp_val;
	Outbuf.last = 0;
	Outbuf.keep = -1;
	stream_out.write(Outbuf);

	tmp_val = stream_in.read();
	AXI_VAL batch_size = tmp_val;
	Outbuf.data = tmp_val;
	Outbuf.last = 0;
	Outbuf.keep = -1;
	stream_out.write(Outbuf);

	tmp_val = stream_in.read();
	AXI_VAL KernelDim = tmp_val;
	Outbuf.data = tmp_val;
	Outbuf.last = 0;
	Outbuf.keep = -1;
	stream_out.write(Outbuf);

	tmp_val = stream_in.read();
	AXI_VAL IFMCH = tmp_val;
	Outbuf.data = tmp_val;
	Outbuf.last = 0;
	Outbuf.keep = -1;
	stream_out.write(Outbuf);

	tmp_val = stream_in.read();
	AXI_VAL IFMDim = tmp_val;
	Outbuf.data = tmp_val;
	Outbuf.last = 0;
	Outbuf.keep = -1;
	stream_out.write(Outbuf);

	tmp_val = stream_in.read();
	AXI_VAL OFMCH = tmp_val;
	Outbuf.data = tmp_val;
	Outbuf.last = 0;
	Outbuf.keep = -1;
	stream_out.write(Outbuf);

	tmp_val = stream_in.read();
	AXI_VAL OFMDim = tmp_val;
	Outbuf.data = tmp_val;
	Outbuf.last = 0;
	Outbuf.keep = -1;
	stream_out.write(Outbuf);

	if (status == 0) // execution
	{
		int OFM_bound = OFMCH * OFMDim * OFMDim * batch_size;
		for (int i = 0; i < OFM_bound; i++)
		{
#pragma HLS loop_tripcount min = 10 max = 1270 avg = 640
#pragma HLS PIPELINE II = 1
			tmp_val = stream_in.read();
			Outbuf.data = tmp_val;
			Outbuf.keep = -1;
			if (i == OFM_bound - 1)
				Outbuf.last = 1;
			else
				Outbuf.last = 0;
			stream_out.write(Outbuf);
		}
	}
	else // weight loading
	{
		int KER_bound = OFMCH * IFMCH * KernelDim * KernelDim;
		for (int i = 0; i < KER_bound; i++)
		{
#pragma HLS loop_tripcount min = 144 max = 147456 avg = 38372
#pragma HLS PIPELINE II = 1
			tmp_val = stream_in.read();
			Outbuf.data = tmp_val;
			Outbuf.keep = -1;
			if (i == KER_bound - 1)
				Outbuf.last = 1;
			else
				Outbuf.last = 0;
			stream_out.write(Outbuf);
		}
	}
}

