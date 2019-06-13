__kernel void blelloch_scan(int n, __global const float * in, __global float * out,  __local float * tmp) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint wgid = get_group_id(0);
    uint bs = get_local_size(0);
    uint d = 1;

    tmp[2 * lid + 0] = 2 * gid + 0 < n ? in[2 * gid + 0] : 0;
    tmp[2 * lid + 1] = 2 * gid + 1 < n ? in[2 * gid + 1] : 0;

    for (uint s = bs; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) {
            uint i = d * (2 * lid + 1) - 1;
            uint j = d * (2 * lid + 2) - 1;
            tmp[j] += tmp[i];
        }
        d <<= 1;
    }
    
    float sum = 0;
    if (lid == 0) {
        sum = tmp[2 * bs - 1];
        tmp[2 * bs - 1] = 0;
    }

    for (uint s = 1; s <= bs; s <<= 1) {
        d >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < s) {
            uint i = d * (2 * lid + 1) - 1;
            uint j = d * (2 * lid + 2) - 1;
            tmp[j] += tmp[i];
            tmp[i] = tmp[j] - tmp[i];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (2 * gid + 0 < n) {
        out[2 * gid + 0] = tmp[2 * lid + 0];
    }
    if (2 * gid + 1 < n) {
        out[2 * gid + 1] = tmp[2 * lid + 1];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        out[2 * bs * wgid] = sum;
    }
}

__kernel void get_parts(int n, __global const float * in, __global float * parts) {
    uint gid = get_global_id(0);
    uint bs = get_local_size(0);
    uint i = 2 * bs * gid;
    if (i < n) {
        parts[gid] = in[i];
    }
}

__kernel void apply_parts(int n, __global float * in, __global const float * parts) {
    uint gid = get_global_id(0);
    uint wgid = get_group_id(0);
    uint bs = get_local_size(0);
    if (2 * gid + 0 < n) {
        if (gid == bs * wgid) {
            in[2 * gid + 0] = 0;
        }
        if (wgid > 0) {
            in[2 * gid + 0] += parts[wgid];
        }
    }
    if (2 * gid + 1 < n && wgid > 0) {
        in[2 * gid + 1] += parts[wgid];
    }
}
