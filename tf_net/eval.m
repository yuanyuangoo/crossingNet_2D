function eval_pck(pred, joints, symmetry_joint_id, joint_name, name)
    % PCK 的实现
    % torso height: || left_shoulder - right hip ||
    % symmetry_joint_id: 具有对称关系的关键点 ID
    % joint_name: 具有对称关系的关键点名字

    range = 0:0.01:0.1;
    show_joint_ids = (symmetry_joint_id >= 1:numel(symmetry_joint_id));

    % compute distance to ground truth joints
    dist = get_dist_pck(pred, joints(1:2, :, :));

    % 计算 PCK
    pck_all = compute_pck(dist, range);
    pck = pck_all(end, :);
    pck(1:end - 1) = (pck(1:end - 1) + pck(symmetry_joint_id)) / 2;

    % 可视化结果
    pck = [pck(show_joint_ids) pck(end)];
    fprintf('------------ PCK Evaluation: %s -------------\n', name);
    fprintf('Parts '); fprintf('& %s ', joint_name{:}); fprintf('& Mean\n');
    fprintf('PCK   '); fprintf('& %.1f  ', pck); fprintf('\n');

    % -------------------------------------------------------------------------
    function dist = get_dist_pck(pred, gt)
        assert(size(pred, 1) == size(gt, 1) && size(pred, 2) == size(gt, 2) && size(pred, 3) == size(gt, 3));

        dist = nan(1, size(pred, 2), size(pred, 3));

        for imgidx = 1:size(pred, 3)
            % torso diameter 躯干直径
            if size(gt, 2) == 14
                refDist = norm(gt(:, 10, imgidx) - gt(:, 3, imgidx));
            elseif size(gt, 2) == 10% 10 joints FLIC
                refDist = norm(gt(:, 7, imgidx) - gt(:, 6, imgidx));
            elseif size(gt, 2) == 11% 11 joints FLIC
                refDist = norm(gt(:, 4, imgidx) - gt(:, 11, imgidx));
            else
                error('Number of joints should be 14 or 10 or 11');
            end

            % 预测的关键点与 gt 关键点的距离
            dist(1, :, imgidx) = sqrt(sum((pred(:, :, imgidx) - gt(:, :, imgidx)).^2, 1)) ./ refDist;

        end

        % -------------------------------------------------------------------------
        function pck = compute_pck(dist, range)
            pck = zeros(numel(range), size(dist, 2) + 1);

            for jidx = 1:size(dist, 2)
                % 计算每个设定阈值的 PCK
                for k = 1:numel(range)
                    pck(k, jidx) = 100 * mean(squeeze(dist(1, jidx, :)) <= range(k));
                end

            end

            % 计算平均 PCK
            for k = 1:numel(range)
                pck(k, end) = 100 * mean(reshape(squeeze(dist(1, :, :)), size(dist, 2) * size(dist, 3), 1) <= range(k));
            end
