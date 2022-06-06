template <
	int InCH,
	int OutCH,
	int UNROLL>
void Fc(
	hls::stream<AXI_VAL> &stream_in,
	hls::stream<AXI_VAL> &stream_out,
	const int layer_id,
	const int output_rectify,
	const int reduce)
{

	static NET_VAL A[InCH], B[OutCH][InCH];

	AXI_VAL Inbuf, Outbuf;

#pragma HLS BIND_STORAGE variable=A type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=B type=ram_2p impl=bram

#pragma HLS ARRAY_PARTITION variable=A dim=1 complete
#pragma HLS ARRAY_PARTITION variable=B cyclic factor=UNROLL/2 dim=1

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
	
	// store weight for current layer
	if (status == layer_id){
		load_weight_OutCH:
		for (int i = 0; i < OutCH; i++){
			load_weight_InCH:
			for (int j = 0; j < InCH; j++){
#pragma HLS PIPELINE II=1
				Inbuf = stream_in.read();
				B[i][j] = Inbuf;
				stream_out.write(B[i][j]);
			}
		}
	}
	
	// execute
	else if (status == 0){

		AXI_CAL buf = 0;

		inference_top:
		for (int num_img = 0; num_img < batch_size; num_img++){
#pragma HLS loop_tripcount min = 1 max = 4 avg = 2
			
			inference_din_InCH:
			for (int i = 0; i < InCH; i++){
#pragma HLS PIPELINE II=1
				Inbuf = stream_in.read();
				A[i] = Inbuf;
			}
			
			inf_out_channel:
			for (int i = 0; i < OutCH; i++){
#pragma HLS UNROLL factor=UNROLL
#pragma HLS loop_tripcount min = 1 max = OutCH avg = OutCH / 2
				buf = 0;
				inf_in_channel:
				for (int j = 0; j < InCH; j++){
#pragma HLS PIPELINE II=1
					buf += A[j] * B[i][j];
				}

				buf = (output_rectify) ? MAX(0, buf) : buf;
				buf >>= reduce;
				Outbuf = buf;
				stream_out.write(Outbuf);
			}
		}
	}
	
	// pass filters for other layers
	else{
		int KER_bound = OFMCH * IFMCH * KernelDim * KernelDim;
		for (int i = 0; i < KER_bound; i++){
#pragma HLS loop_tripcount min=144 max=147456 avg=38372
#pragma HLS PIPELINE II=1
			Inbuf = stream_in.read();
			stream_out.write(Inbuf);
		}
	}
}
