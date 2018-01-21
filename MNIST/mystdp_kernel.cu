__global__ void stdp_kernel(
            float *weight, int weight_size_0, int weight_size_1, int weight_size_2, int weight_size_3, 
            float *output_spike, int output_spike_size_0, int output_spike_size_1, 
                                 int output_spike_size_2, int output_spike_size_3,
            int *Ron, int *Roff, float *v, float *m, float *n, float *p, float *q,
            float *pp, float *qq, long *a, long *b, float *dt,
            float *history, float *weight_update, f())
{
 
    // Each thread is for an element in output_spike in the form:
    // (batch_id, channel_id, height_id, width_id) = (blockIdx.x, threadIdx.x, blockIdx.y, blockIdx.z).
    // Note that grid is 3D and block is 1D; this to make the channel the fastest dimension as across 
    // the channel the history patch block is accessed is same.   
    // IMPORTANT: Number of channels MUST BE LESS than maximum thread limit.
    // Note: All 'id's refer to numpy indices and 'index'/'linear_idx' refer to linear memory index.

    const int offset = weight_size_2 / 2;
    const int batch_id = blockIdx.x;
    const int channel_id = threadIdx.x;
    const int height_id = blockIdx.y;
    const int width_id = blockIdx.z;

    /*
    // some constants
    output_spike_size_1_2_3 = output_spike_size_1 * output_spike_size_2 * output_spike_size_3;
    output_spike_size_2_3 = output_spike_size_2 * output_spike_size_3;
    output_spike_size_3 = output_spike_size_3; // = gridDim.z;
    weight_size_0_1_2_3 = weight_size_0 * weight_size_1 * weight_size_2 * weight_size_3;
    weight_size_1_2_3 = weight_size_1 * weight_size_2 * weight_size_3;
    weight_size_2_3 = weight_size_2 * weight_size_3;
    weight_size_1 = weight_size_1
    weight_size_3 = weight_size_3
    */

    const int linear_idx = (batch_id * (output_spike_size_1 * output_spike_size_2 * output_spike_size_3)) \
                            + (channel_id * (output_spike_size_2 * output_spike_size_3)) \
                            + (height_id * (output_spike_size_3)) \
                            + width_id;

    if (output_spike[linear_idx] != 1.0f)
        return;

    const int filter_index = channel_id * (weight_size_1 * weight_size_2 * weight_size_3);
    const int delta_index_const = batch_id * (weight_size_0 * weight_size_1 * weight_size_2 * weight_size_3);

    float input_tmp;
    int w_index, deltaW_index;
    int filter_2d_index, filter_1d_index;
    int history_1d_index, history_2d_index, history_index, history_index1, history_index2;
    double mu_v,R_on,R,para
    short p1,p2,s1,s2
    fp() = f();

    for (int l = 0; l < weight_size_1; l++)
    {
        history_2d_index = (batch_id * (weight_size_1 * output_spike_size_2 * output_spike_size_3)) \
                        + (l * (output_spike_size_2 * output_spike_size_3));
        filter_2d_index = filter_index + (l * weight_size_2 * weight_size_3);

        for (int i = 0; i < weight_size_2; i++)
        {
            filter_1d_index = filter_2d_index + (i * weight_size_3);
            history_index1 = height_id + i - offset;
            if (history_index1 < 0 || history_index1 >= output_spike_size_2)
                continue;
            history_1d_index = history_2d_index + (history_index1 * output_spike_size_3);

            for (int j = 0; j < weight_size_3; j++)
            {
                w_index = filter_1d_index + j;
                deltaW_index = delta_index_const + w_index;
                history_index2 = width_id + j - offset;
                if (history_index2 < 0 || history_index2 >= output_spike_size_3)
                    continue;
                history_index = history_1d_index + history_index2;
                input_tmp = history[history_index];

                weight_update[deltaW_index] = weight[deltaW_index]*weight[deltaW_index]*dt[deltaW_index]*\
                              a[deltaW_index]*(v[deltaW_index]/((Roff[deltaW_index]-Ron[deltaW_index])/ \
                              weight[deltaW_index]+Ron[deltaW_index])-m[deltaW_index]) \
                              *(Roff[deltaW_index]-Ron[deltaW_index])*(p[deltaW_index]-(1-1/weight[deltaW_index])/ \
                              (1-p[deltaW_index])+pp[deltaW_index])*(p[deltaW_index]-(1-1/weight[deltaW_index])/ \
                              (1-p[deltaW_index])+pp[deltaW_index])/(1- dt[deltaW_index]*\
                              a[deltaW_index]*(v[deltaW_index]/((Roff[deltaW_index]-Ron[deltaW_index])/ \
                              weight[deltaW_index]+Ron[deltaW_index])-m[deltaW_index]) \
                              *(Roff[deltaW_index]-Ron[deltaW_index])*(p[deltaW_index]-(1-1/weight[deltaW_index])/ \
                              (1-p[deltaW_index])+pp[deltaW_index])*(p[deltaW_index]-(1-1/weight[deltaW_index])/ \
                              (1-p[deltaW_index])+pp[deltaW_index]))*(input_tmp != 0.0f);

                weight_update[deltaW_index] = weight[deltaW_index]*weight[deltaW_index]*dt[deltaW_index]*\
                              b[deltaW_index]*(v[deltaW_index]/((Roff[deltaW_index]-Ron[deltaW_index])/ \
                              weight[deltaW_index]+Ron[deltaW_index])-n[deltaW_index]) \
                              *(Roff[deltaW_index]-Ron[deltaW_index])*(q[deltaW_index]-(1-1/weight[deltaW_index])/ \
                              (1-q[deltaW_index])-qq[deltaW_index])*(q[deltaW_index]-(1-1/weight[deltaW_index])/ \
                              (1-q[deltaW_index])-qq[deltaW_index])*(q[deltaW_index]-(1-1/weight[deltaW_index])/ \
                              (1-q[deltaW_index])-qq[deltaW_index])/(1-dt[deltaW_index]*\
                              b[deltaW_index]*(v[deltaW_index]/((Roff[deltaW_index]-Ron[deltaW_index])/ \
                              weight[deltaW_index]+Ron[deltaW_index])-n[deltaW_index]) \
                              *(Roff[deltaW_index]-Ron[deltaW_index])*(q[deltaW_index]-(1-1/weight[deltaW_index])/ \
                              (1-q[deltaW_index])-qq[deltaW_index])*(q[deltaW_index]-(1-1/weight[deltaW_index])/ \
                              (1-q[deltaW_index])-qq[deltaW_index])*(q[deltaW_index]-(1-1/weight[deltaW_index])/ \
                              (1-q[deltaW_index])-qq[deltaW_index]))*(input_tmp == 0.0f);

            }
        }
    }
}


