function C = dvsMc(X, mix_step, k)
    [n,m] = size(X);
    assert(k > n);
    assert(m >= k);
    init = randperm(m);
    C = init(1:k);
    C_bar = init(k+1:end);
    
    M = X(:,C) * X(:,C)';
    M_inv = pinv(M);
    M_det = det(M);
    
    for i = 1:mix_step
        idx_in_pos = randi(length(C_bar));
        idx_out_pos = randi(length(C));
        idx_in = C_bar(idx_in_pos);
        idx_out = C(idx_out_pos);
        vec_in = X(:,idx_in);
        vec_out = X(:,idx_out);
        
        tmp = vec_out' * M_inv * vec_out;
        inter_inv = M_inv + (M_inv * vec_out * (vec_out' * M_inv)) / (1 - tmp);
        inter_det = (1 - tmp) * M_det;
        
        tmp = vec_in' * inter_inv * vec_in;
        update_inv = inter_inv - (inter_inv * vec_in * (vec_in' * inter_inv)) / (1 + tmp);
        update_det = (1 + tmp) * inter_det;
        
        if rand < update_det / M_det
            M_det = update_det;
            M_inv = update_inv;
            C_bar(idx_in_pos) = idx_out;
            C(idx_out_pos) = idx_in;
        end 
    end
end