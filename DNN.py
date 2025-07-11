import tensorflow as tf

def shifted_normal_tf(x):
    # 1) NumPy로 미리 구해둔 값 (예시)
    # mu_val = 0.6737 - 좌우
    # M_val  = 0.1034 - 상하
    # sigma = 0.7 - 표준편차 (클수록 완만하게 퍼짐)
    # base=0.001 - x→∞일 때 수렴할 기저값
    x = tf.clip_by_value(x, 0, 2)
    mu_val = 0.7
    M_val  = 0.125
    sigma  = 0.575
    base   = 0.01
    # Tensor 상수 선언
    mu_tf    = tf.constant(mu_val, dtype=tf.float32)
    M_tf     = tf.constant(M_val, dtype=tf.float32)
    sigma_tf = tf.constant(sigma, dtype=tf.float32)
    base_tf  = tf.constant(base, dtype=tf.float32)

    # 가우시안 형태
    sq   = tf.square(x - mu_tf)
    expo = tf.exp(-0.5 * sq / (sigma_tf**2))

    # peak에서 base로 부드럽게 이동
    return (M_tf - base_tf) * expo + base_tf

def smooth_labels(one_hot_labels, num_classes, alpha=0.1):
    """
    one_hot_labels: shape = (batch_size, num_classes), dtype=tf.float32
    num_classes: 정수
    alpha: smoothing 계수 (예: 0.1)
    returns: smoothed one-hot vector
    """
    # 예: α=0.1, num_classes=6이면
    #  one-hot = [0,0,1,0,0,0] → [0.1/6, 0.1/6, 0.9, 0.1/6, 0.1/6, 0.1/6]
    smooth_pos = 1.0 - alpha
    smooth_neg = alpha / tf.cast(num_classes, tf.float32)
    return one_hot_labels * smooth_pos + smooth_neg

def leaky_relu_derivative(x, alpha=0.01):
    return tf.where(x >= 0, tf.ones_like(x), tf.fill(tf.shape(x), alpha))

def ret_min_value(x) :
    return tf.math.exp(x / (1 - x))

def diff_fuc(x, alpha, init_val) :
    return tf.math.log(init_val + alpha * x) / (1 + tf.math.log(init_val + alpha * x))

class DNN:
    def __init__(self, layer_sizes, batch_size=128, initializer='xavier', dropout_rate=0.004, label_smoothing=0.025):
        self.layer_sizes = layer_sizes
        self.initializer = initializer
        self.batch_size = batch_size
        self.weight_szie = len(layer_sizes) - 1
        self.weights = []  # W₁, W₂, ..., Wₙ
        self.biases = []   # b₁, b₂, ..., bₙ
        self.mW = []
        self.mb = []
        self.loss = []
        self.val_loss = []
        self._initialize_weights()
        self.square = 7
        self.dropout_rate  = dropout_rate  # 드롭아웃 비율
        self.label_smoothing  = label_smoothing

    def _get_initializer(self, fan_in):
        if self.initializer == 'he':
            stddev = tf.math.sqrt(2.0 / tf.cast(fan_in, tf.float32))
        elif self.initializer == 'xavier':
            stddev = tf.math.sqrt(1.0 / tf.cast(fan_in, tf.float32))
        else:
            stddev = tf.constant(0.01, dtype=tf.float32)
        return stddev

    def _initialize_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            in_dim = self.layer_sizes[i]
            out_dim = self.layer_sizes[i + 1]
            stddev = self._get_initializer(in_dim)

            W = tf.Variable(
                tf.random.normal(shape=(out_dim, in_dim), stddev=stddev),
                trainable=True
            )
            b = tf.Variable(tf.zeros(shape=(out_dim, 1)), trainable=True)

            self.weights.append(W)
            self.biases.append(b)
            self.mW.append(tf.Variable(tf.zeros_like(W), trainable=False))
            self.mb.append(tf.Variable(tf.zeros_like(b), trainable=False))

    
    @tf.function(jit_compile=True)
    def _forward(self, X_input, training=True):
        """
        X_input:  shape=(input_dim, batch_size) 텐서
        training: True일 때만 은닉층에 Dropout을 적용
        """
        X = X_input  # shape = (input_dim, batch_size)

        X_list = []
        Z_list = []
        y_pred = None

        weights_size = self.weight_szie
        for idx in range(weights_size):
            W = self.weights[idx]   # shape = (units, prev_units)
            b = self.biases[idx]    # shape = (units, 1)

            # (1) 선형 계산
            Z = tf.matmul(W, X) + b  # shape = (units, batch_size)

            X_list.append(X)
            Z_list.append(Z)

            if idx < weights_size - 1:
                # (2) 은닉층: LeakyReLU → Dropout (training=True일 때만)
                H = tf.nn.leaky_relu(Z, alpha=0.01)  # shape = (units, batch_size)

                if training:
                    # tf.nn.dropout은 자동으로 keep_prob = 1 - rate 만큼 스케일 보정해 줌
                    H = tf.nn.dropout(H, rate=self.dropout_rate)

                X = H
            else:
                # (3) 출력층: softmax
                y_pred = tf.nn.softmax(Z, axis=0)  # axis=0: units 축 기준 softmax

        return X_list, Z_list, y_pred
    
    @tf.function(jit_compile=True)
    def _tensorOfH(self, idx, weights_size, J, D, Je, last_w, cur_w, y_pred, Z_list) :
        last_weight = None
        if idx > 0 :
            J = tf.matmul(cur_w, J, transpose_a=True) * leaky_relu_derivative(Z_list[idx - 1])
        if idx == weights_size - 1 :
            T = y_pred - tf.square(y_pred)
            last_weight = cur_w
        elif idx == weights_size - 2 :
            term1 = tf.matmul(last_w ** 2, y_pred, transpose_a=True)
            term2 = tf.square(tf.matmul(last_w, y_pred, transpose_a=True))
            T = (term1 - term2) * tf.square(leaky_relu_derivative(Z_list[idx]))
            last_weight = cur_w
        else:
            # Wt: [out_dim, in_dim]
            Wt = tf.transpose(last_w)  
            # dZ: [out_dim, batch_size] 혹은 [out_dim, m]
            dZ = leaky_relu_derivative(Z_list[idx])
            tmp = tf.tensordot(Wt,                # [..., s]
                            D,                 # [   s, j, k]
                            axes=[[1], [0]])   # -> [a, j, k]
            # 이제 tmp[a, j, k] 와 Wt[a, j] 를 곱하고 j축(=axis=1)으로 합치면 [a, k]
            term1 = tf.reduce_sum(tmp * tf.expand_dims(Wt, -1), axis=1)
            # term2는 그대로 matmul
            # term2[a, m] = sum_j Wt[a,j] * Je[j,m]
            term2 = tf.matmul(Wt, Je)  # [out_dim, m]
            # T[a, k] 계산
            T = (term1 - tf.square(term2)) * tf.square(dZ)
            # --- D 업데이트: einsum('ij,kl,jlm,im,km->ikm') 대체 ---
            # D_old[s, s, m] -> batch 형식 [m, s, s]
            D_b = tf.transpose(D, perm=[2, 0, 1])      # [m, s, s]
            # Wt를 batch 차원에 맞춰 확장
            Wt_b = tf.expand_dims(Wt, 0)               # [1, out, in]
            # 1) batched matmul: Wt_b @ D_b -> [m, out, in]
            mid = tf.matmul(Wt_b, D_b)
            # 2) batched matmul: mid @ Wt^T -> [m, out, out]
            WtT_b = tf.expand_dims(tf.transpose(Wt), 0)  # [1, in, out]
            M_b = tf.matmul(mid, WtT_b)                   # [m, out, out]
            # 3) 차원 재배치: [out, out, m]
            M = tf.transpose(M_b, perm=[1, 2, 0])         # [out, out, m]
            # 4) 마지막으로 dZ[i,m] * dZ[k,m] 브로드캐스트 곱
            #    dZ_expand_i: [out, 1, m], dZ_expand_k: [1, out, m]
            D = M * tf.expand_dims(dZ, 1) * tf.expand_dims(dZ, 0)  # [out, out, m]
            # Je 업데이트
            Je = term2 * dZ
            last_weight = cur_w
        return T, J, D, Je, last_weight
    
    @tf.function(jit_compile=True)
    def _Hsolve(self, T, nX, dP, current_batch_size, coef_outerDp, coef_H) :
        lamb=1e-2
        H = tf.matmul(T, tf.square(nX), transpose_b=True) / current_batch_size
        outer_dP = tf.square(dP)
        newH = coef_outerDp * outer_dP + coef_H * H
        ##################################################
        newH = tf.abs(newH)
        newH = newH + tf.fill(newH.shape, lamb)
        L = dP * coef_H / newH
        return L, lamb
    
    @tf.function(jit_compile=True)
    def _hessian_block_step(self, X_batch, y_batch, epoch) :
        square = self.square
        _tensorOfH = self._tensorOfH
        _Hsolve = self._Hsolve
        weights_size = self.weight_szie
        weights = self.weights
        biases = self.biases
        mW = self.mW
        mb = self.mb

        last_weight = None
        current_batch_size = tf.cast(tf.shape(X_batch)[1], tf.float32)
        X_list, Z_list, y_pred = self._forward(X_batch)
        y_batch = smooth_labels(y_batch, tf.shape(y_batch)[0], alpha=self.label_smoothing)
        J = y_pred - y_batch
        WT = tf.transpose(self.weights[weights_size - 1])
        WT_outer = WT[:, None, :] * WT[None, :, :]     # shape: [I, I, S]
        D = tf.tensordot(WT_outer, y_pred, axes=[[2], [0]])
        Je = tf.matmul(WT, y_pred) * leaky_relu_derivative(Z_list[weights_size - 2])
        loss = -tf.reduce_mean(tf.reduce_sum(y_batch * tf.math.log(y_pred + 1e-8), axis=0))
        diff = 0.25
        ####diff_coef#####
        coef_outerDp = square * (square -  1) * (diff) ** (square - 2)
        coef_H = square * (diff) ** (square - 1)
        ##################
        for idx in reversed(range(weights_size)):
            T = None
            cur_w = weights[idx]
            cur_b = biases[idx]
            n = tf.shape(cur_w)[1]
            ones = tf.ones((1, tf.shape(X_list[idx])[1]), dtype=X_list[idx].dtype)
            nX = tf.concat([X_list[idx], ones], axis=0)
            dW = tf.matmul(J, X_list[idx], transpose_b=True) / current_batch_size
            db = tf.reduce_sum(J, axis=1, keepdims=True) / current_batch_size
            dP = tf.concat([dW, db], axis=1)
            T, J, D, Je, last_weight = _tensorOfH(idx, weights_size, J, D, Je, last_weight, cur_w, y_pred, Z_list)
            L, lamb = _Hsolve(T, nX, dP, current_batch_size, coef_outerDp, coef_H)
            d2W = L[:, :n]
            d2b = L[:, n:n+1]
            #######lr_damping#######
            damp = shifted_normal_tf(loss)
            lr = 2 * lamb / (10.0) * damp / diff ** (square-1)
            ########################
            beta1 = 0.25
            mW[idx].assign((beta1) * mW[idx] + (1 - beta1) * d2W)
            mb[idx].assign((beta1) * mb[idx] + (1 - beta1) * d2b)
            mW_hat = mW[idx] / (1 - beta1 ** epoch)
            mb_hat = mb[idx] / (1 - beta1 ** epoch)

            weights[idx].assign(cur_w - lr * mW_hat)
            biases[idx].assign(cur_b - lr * mb_hat)

    @tf.function(jit_compile=True)
    def _train_step(self, X_batch, y_batch, epoch):
        self._hessian_block_step(tf.transpose(X_batch), tf.transpose(y_batch), tf.cast(epoch+1, tf.float32))

    def training(self, X, y_onehot, X_val, y_val_onehot, epochs=5):
        N = tf.cast(tf.shape(X)[0], tf.int64)
        dataset = tf.data.Dataset.from_tensor_slices((X, y_onehot))
        dataset = dataset.shuffle(buffer_size=N, reshuffle_each_iteration=True).batch(self.batch_size).repeat()  
        steps_per_epoch = tf.cast(tf.math.ceil(N / self.batch_size), tf.int64)
        loss_list = self.loss
        val_loss_list = self.val_loss
        for epoch in tf.range(epochs):
            # X_batch shape: [input_dim, batch_size]
            # y_batch: [output_dim, batch_size]
            for X_batch, y_batch in dataset.take(steps_per_epoch):
                self._train_step(X_batch, y_batch, epoch)
            if ((epoch + 1) % 5 == 0 or epoch == 0) :
                loss = self.compute_loss(X, y_onehot)
                val_loss = self.compute_loss(X_val, y_val_onehot)
                loss_list.append(loss)
                val_loss_list.append(val_loss)

                tf.print(f'custom {epoch+1} epoch : training loss : {loss} - val loss : {val_loss}')
    
    def ret_weights_biases(self) :
        return self.weights, self.biases

    def compute_loss(self, x, y_onehot):
        X = x  # shape: (batch_size, input_dim)

        for idx in range(len(self.weights)):
            W = tf.transpose(self.weights[idx])  # transpose to (in_dim, out_dim)
            Z = tf.matmul(X, W) + tf.reshape(self.biases[idx], (1, -1))  # shape: (batch_size, out_dim)
            if idx < len(self.weights) - 1:
                X = tf.nn.leaky_relu(Z, alpha=0.01)
            else:
                logits = Z

        y_pred = tf.nn.softmax(logits, axis=1)
        loss = tf.reduce_mean(tf.reduce_sum(-y_onehot * tf.math.log(y_pred + 1e-8), axis=1))
        return loss

    
    def ret_loss_list(self) :
        return self.loss
    
    def ret_val_loss_list(self) :
        return self.val_loss

