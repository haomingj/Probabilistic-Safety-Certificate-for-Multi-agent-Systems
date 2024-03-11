#define PI 3.141592654f

__global__ void dynamic_evolution(float *state, float *goal, float *noise, float *allstate, float *parameters){
    float DT = parameters[0];
    float DX1 = parameters[1];
    float DX2 = parameters[2];
    float DX3 = parameters[3];
    float DX4 = parameters[4];
    float MIN_DIST = parameters[5];
    int T = lrintf(parameters[6]);
    int K = lrintf(parameters[7]);
    float GAIN = parameters[8];
    float VMAX = parameters[9];
    float SIGMA1 = parameters[10];
    float SIGMA2 = parameters[11];
    if (blockIdx.x == 4*blockIdx.y){
        if (threadIdx.x < blockDim.x/4){
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T] = state[4*blockIdx.y] - DX1;
        } else if (threadIdx.x < blockDim.x/2){
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T] = state[4*blockIdx.y] - 2.0f*DX1;
        } else if (threadIdx.x < 3*blockDim.x/4){
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T] = state[4*blockIdx.y] - 3.0f*DX1;
        } else {
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T] = state[4*blockIdx.y] - 4.0f*DX1;
        }
    } else if (blockIdx.x == 4*blockIdx.y + 4*gridDim.y){
        if (threadIdx.x < blockDim.x/4){
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T] = state[4*blockIdx.y] + DX1;
        } else if (threadIdx.x < blockDim.x/2){
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T] = state[4*blockIdx.y] + 2.0f*DX1;
        } else if (threadIdx.x < 3*blockDim.x/4){
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T] = state[4*blockIdx.y] + 3.0f*DX1;
        } else {
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T] = state[4*blockIdx.y] + 4.0f*DX1;
        }
    } else {
        allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T] = state[4*blockIdx.y];
    }
    if (blockIdx.x == 4*blockIdx.y + 1){
        if (threadIdx.x < blockDim.x/4){
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T] = state[4*blockIdx.y+1] - DX2;
        } else if (threadIdx.x < blockDim.x/2){
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T] = state[4*blockIdx.y+1] - 2.0f*DX2;
        } else if (threadIdx.x < 3*blockDim.x/4){
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T] = state[4*blockIdx.y+1] - 3.0f*DX2;
        } else {
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T] = state[4*blockIdx.y+1] - 4.0f*DX2;
        }
    } else if (blockIdx.x == 4*blockIdx.y + 1 + 4*gridDim.y){
        if (threadIdx.x < blockDim.x/4){
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T] = state[4*blockIdx.y+1] + DX2;
        } else if (threadIdx.x < blockDim.x/2){
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T] = state[4*blockIdx.y+1] + 2.0f*DX2;
        } else if (threadIdx.x < 3*blockDim.x/4){
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T] = state[4*blockIdx.y+1] + 3.0f*DX2;
        } else {
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T] = state[4*blockIdx.y+1] + 4.0f*DX2;
        }
    } else {
        allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T] = state[4*blockIdx.y+1];
    }
    float v;
    float theta;
    if (blockIdx.x == 4*blockIdx.y + 2){
        if (threadIdx.x < blockDim.x/4){
            v = state[4*blockIdx.y+2] - DX3;
        } else if (threadIdx.x < blockDim.x/2){
            v = state[4*blockIdx.y+2] - 2.0f*DX3;
        } else if (threadIdx.x < 3*blockDim.x/4){
            v = state[4*blockIdx.y+2] - 3.0f*DX3;
        } else {
            v = state[4*blockIdx.y+2] - 4.0f*DX3;
        }
    } else if (blockIdx.x == 4*blockIdx.y + 2 + 4*gridDim.y){
        if (threadIdx.x < blockDim.x/4){
            v = state[4*blockIdx.y+2] + DX3;
        } else if (threadIdx.x < blockDim.x/2){
            v = state[4*blockIdx.y+2] + 2.0f*DX3;
        } else if (threadIdx.x < 3*blockDim.x/4){
            v = state[4*blockIdx.y+2] + 3.0f*DX3;
        } else {
            v = state[4*blockIdx.y+2] + 4.0f*DX3;
        }
    } else {
        v = state[4*blockIdx.y+2];
    }
    if (blockIdx.x == 4*blockIdx.y + 3){
        if (threadIdx.x < blockDim.x/4){
            theta = state[4*blockIdx.y+3] - DX4;
        } else if (threadIdx.x < blockDim.x/2){
            theta = state[4*blockIdx.y+3] - 2.0f*DX4;
        } else if (threadIdx.x < 3*blockDim.x/4){
            theta = state[4*blockIdx.y+3] - 3.0f*DX4;
        } else {
            theta = state[4*blockIdx.y+3] - 4.0f*DX4;
        }
    } else if (blockIdx.x == 4*blockIdx.y + 3 + 4*gridDim.y){
        if (threadIdx.x < blockDim.x/4){
            theta = state[4*blockIdx.y+3] + DX4;
        } else if (threadIdx.x < blockDim.x/2){
            theta = state[4*blockIdx.y+3] + 2.0f*DX4;
        } else if (threadIdx.x < 3*blockDim.x/4){
            theta = state[4*blockIdx.y+3] + 3.0f*DX4;
        } else {
            theta = state[4*blockIdx.y+3] + 4.0f*DX4;
        }
    } else {
        theta = state[4*blockIdx.y+3];
    }
    if (v < 0.0f){
        v = -v;
        theta = theta - PI;
    }
    if (theta > PI){
        theta = theta - 2*PI;
    } else if (theta < -PI){
        theta = theta + 2*PI;
    }
    float theta_ref;
    for (int i = 0; i < T-1; i++){
        allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T+i+1] = 
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T+i] + v*cosf(theta)*DT;
        allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T+i+1] =
            allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T+i] + v*sinf(theta)*DT;
        v = v + GAIN*(VMAX-v)*DT + SIGMA1 * noise[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T+i]*DT;
        if (v < 0.0f) {
            v = -v;
            theta = theta - PI;
        }
        if (theta > PI){
            theta = theta - 2*PI;
        } else if (theta < -PI){
            theta = theta + 2*PI;
        }
        theta_ref = atan2f(goal[2*blockIdx.y+1]-allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T+i],
                             goal[2*blockIdx.y]-allstate[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T+i]);
        if (fabsf(theta_ref-theta) < PI){
            theta = theta + GAIN*(theta_ref-theta)*DT + SIGMA2 * noise[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T+i]*DT;
        } else {
            theta = theta + GAIN*(theta-theta_ref)*DT + SIGMA2 * noise[blockIdx.x*blockDim.x*2*gridDim.y*T+threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T+i]*DT;
        }
        
    }
    return;
}

__global__ void identify_collision(float *allstate, float *collision, float *temp_state, float *parameters){
    float DT = parameters[0];
    float DX1 = parameters[1];
    float DX2 = parameters[2];
    float DX3 = parameters[3];
    float DX4 = parameters[4];
    float MIN_DIST = parameters[5];
    int T = lrintf(parameters[6]);
    int K = lrintf(parameters[7]);
    float GAIN = parameters[8];
    float VMAX = parameters[9];
    float SIGMA1 = parameters[10];
    float SIGMA2 = parameters[11];
    collision[blockIdx.x*blockDim.x*gridDim.y+threadIdx.x*gridDim.y+blockIdx.y] = 0.0f;
    int center_agent;
    if (blockIdx.x < gridDim.x/2){
        center_agent = blockIdx.x/4;
    } else {
        center_agent = (blockIdx.x-gridDim.x/2)/4;
    }
    // test
    // center_agent = 0;
    if (blockIdx.y == 1 && threadIdx.x == 0 && blockIdx.x == 0){
        for (int i = 0; i < gridDim.x/8; i++) {
            temp_state[2*i] = allstate[blockIdx.x*blockDim.x*gridDim.x/4*gridDim.y+threadIdx.x*gridDim.x/4*gridDim.y+2*i*gridDim.y+blockIdx.y];
            temp_state[2*i+1] = allstate[blockIdx.x*blockDim.x*gridDim.x/4*gridDim.y+threadIdx.x*gridDim.x/4*gridDim.y+(2*i+1)*gridDim.y+blockIdx.y];
        }
    }
    float xref = allstate[blockIdx.x*blockDim.x*gridDim.x/4*gridDim.y+threadIdx.x*gridDim.x/4*gridDim.y+2*center_agent*gridDim.y+blockIdx.y];
    float yref = allstate[blockIdx.x*blockDim.x*gridDim.x/4*gridDim.y+threadIdx.x*gridDim.x/4*gridDim.y+(2*center_agent+1)*gridDim.y+blockIdx.y];
    float x;
    float y;
    for (int i = 0; i < gridDim.x/8; i++) {
        if (i == center_agent){
            continue;
        }
        x = allstate[blockIdx.x*blockDim.x*gridDim.x/4*gridDim.y+threadIdx.x*gridDim.x/4*gridDim.y+2*i*gridDim.y+blockIdx.y];
        y = allstate[blockIdx.x*blockDim.x*gridDim.x/4*gridDim.y+threadIdx.x*gridDim.x/4*gridDim.y+(2*i+1)*gridDim.y+blockIdx.y];
        if ((x-xref)*(x-xref)+(y-yref)*(y-yref) < MIN_DIST*MIN_DIST){
            collision[blockIdx.x*blockDim.x*gridDim.y+threadIdx.x*gridDim.y+blockIdx.y] = 1.0f;
            return;
        }
    }
    collision[blockIdx.x*blockDim.x*gridDim.y+threadIdx.x*gridDim.y+blockIdx.y] = 0.0f;
    return;
}

__global__ void identify_safety(float *collision, float *safety, float *parameters){
    float DT = parameters[0];
    float DX1 = parameters[1];
    float DX2 = parameters[2];
    float DX3 = parameters[3];
    float DX4 = parameters[4];
    float MIN_DIST = parameters[5];
    int T = lrintf(parameters[6]);
    int K = lrintf(parameters[7]);
    float GAIN = parameters[8];
    float VMAX = parameters[9];
    float SIGMA1 = parameters[10];
    float SIGMA2 = parameters[11];
    for (int i = 1; i < T; i++) {
        if (collision[blockIdx.x*blockDim.x*T+threadIdx.x*T+i] >= 0.5f){
            safety[blockIdx.x*blockDim.x+threadIdx.x] = 0.0f;
            return;
        }
    }
    safety[blockIdx.x*blockDim.x+threadIdx.x] = 1.0f;
    return;
}

__global__ void dynamic_evolution_no_gradient(float *state, float *goal, float *noise, float *allstate, float *parameters){
    float DT = parameters[0];
    float DX1 = parameters[1];
    float DX2 = parameters[2];
    float DX3 = parameters[3];
    float DX4 = parameters[4];
    float MIN_DIST = parameters[5];
    int T = lrintf(parameters[6]);
    int K = lrintf(parameters[7]);
    float GAIN = parameters[8];
    float VMAX = parameters[9];
    float SIGMA1 = parameters[10];
    float SIGMA2 = parameters[11];
    allstate[threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T] = state[4*blockIdx.y];
    allstate[threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T] = state[4*blockIdx.y+1];
    float v = state[4*blockIdx.y+2];
    float theta = state[4*blockIdx.y+3];
    float theta_ref;
    for (int i = 0; i < T-1; i++){
        allstate[threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T+i+1] = allstate[threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T+i] + v*cosf(theta)*DT;
        allstate[threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T+i+1] = allstate[threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T+i] + v*sinf(theta)*DT;
        v = v + GAIN*(VMAX-v)*DT + SIGMA1 * noise[threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T+i];
        theta_ref = atan2f(goal[2*blockIdx.y+1]-allstate[threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T+i],
                             goal[2*blockIdx.y]-allstate[threadIdx.x*2*gridDim.y*T+2*blockIdx.y*T+i]);
        if (fabsf(theta_ref-theta) < PI){
            theta = theta + GAIN*(theta_ref-theta)*DT + SIGMA2 * noise[threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T+i];
        } else {
            theta = theta + GAIN*(theta-theta_ref)*DT + SIGMA2 * noise[threadIdx.x*2*gridDim.y*T+(2*blockIdx.y+1)*T+i];
        }
    }
    return;
}

__global__ void identify_collision_no_gradient(float *allstate, float *collision, float *parameters){
    float DT = parameters[0];
    float DX1 = parameters[1];
    float DX2 = parameters[2];
    float DX3 = parameters[3];
    float DX4 = parameters[4];
    float MIN_DIST = parameters[5];
    int T = lrintf(parameters[6]);
    int K = lrintf(parameters[7]);
    float GAIN = parameters[8];
    float VMAX = parameters[9];
    float SIGMA1 = parameters[10];
    float SIGMA2 = parameters[11];
    collision[blockIdx.x*blockDim.x*gridDim.y+threadIdx.x*gridDim.y+blockIdx.y] = 0.0f;
    int center_agent = blockIdx.x;
    float xref = allstate[threadIdx.x*gridDim.x*2*gridDim.y+2*center_agent*gridDim.y+blockIdx.y];
    float yref = allstate[threadIdx.x*gridDim.x*2*gridDim.y+(2*center_agent+1)*gridDim.y+blockIdx.y];
    float x;
    float y;
    for (int i = 0; i < gridDim.x; i++) {
        if (i == center_agent){
            continue;
        }
        x = allstate[threadIdx.x*gridDim.x*2*gridDim.y+2*i*gridDim.y+blockIdx.y];
        y = allstate[threadIdx.x*gridDim.x*2*gridDim.y+(2*i+1)*gridDim.y+blockIdx.y];
        if (sqrtf((x-xref)*(x-xref)+(y-yref)*(y-yref)) < MIN_DIST){
            collision[blockIdx.x*blockDim.x*gridDim.y+threadIdx.x*gridDim.y+blockIdx.y] = 1.0f;
            return;
        }
    }
    collision[blockIdx.x*blockDim.x*gridDim.y+threadIdx.x*gridDim.y+blockIdx.y] = 0.0f;
    return;
}

__global__ void identify_safety_along_n(float *n_agent, float *collision, float *collision_n, float *parameters){
    float DT = parameters[0];
    float DX1 = parameters[1];
    float DX2 = parameters[2];
    float DX3 = parameters[3];
    float DX4 = parameters[4];
    float MIN_DIST = parameters[5];
    int T = lrintf(parameters[6]);
    int K = lrintf(parameters[7]);
    float GAIN = parameters[8];
    float VMAX = parameters[9];
    float SIGMA1 = parameters[10];
    float SIGMA2 = parameters[11];
    int nagent = lrintf(n_agent[0]);
    for (int i = 0; i < nagent; i++) {
        if (collision[i*blockDim.x*gridDim.x+threadIdx.x*gridDim.x+blockIdx.x] >= 0.5f){
            collision_n[blockIdx.x*blockDim.x+threadIdx.x] = 1.0f;
            return;
        }
    }
    collision_n[blockIdx.x*blockDim.x+threadIdx.x] = 0.0f;
    return;
}

__global__ void identify_safety_along_T(float *collision_n, float *safety, float *parameters){
    float DT = parameters[0];
    float DX1 = parameters[1];
    float DX2 = parameters[2];
    float DX3 = parameters[3];
    float DX4 = parameters[4];
    float MIN_DIST = parameters[5];
    int T = lrintf(parameters[6]);
    int K = lrintf(parameters[7]);
    float GAIN = parameters[8];
    float VMAX = parameters[9];
    float SIGMA1 = parameters[10];
    float SIGMA2 = parameters[11];
    for (int i = 1; i < T; i++) {
        if (collision_n[i*blockDim.x+threadIdx.x] >= 0.5f){
            safety[threadIdx.x] = 0.0f;
            return;
        }
    }
    safety[threadIdx.x] = 1.0f;
    return;
}