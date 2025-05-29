# utils/unify.py
import os
import re
import csv
from collections import defaultdict


def process_metrics_to_csv(path_a):
    # 数据结构：dict[test编号][时间][指标][route] = 值
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    test_routes = defaultdict(set)  # 记录每个test包含的route
    test_time_metrics = defaultdict(lambda: defaultdict(set))  # 记录每个test的时间与指标

    # 遍历所有txt文件
    for filename in os.listdir(path_a):
        if not filename.endswith(".txt") or not filename.startswith("metrics_"):
            continue

        # 解析文件名
        match = re.match(r"metrics_route(\d+)_(test\d+)_(\d+)min\.txt", filename)
        if not match:
            continue

        route_num, test_num, time = match.groups()
        route_key = f"route{route_num}"
        time_key = f"{time}min"

        # 读取指标数据
        with open(os.path.join(path_a, filename), 'r') as f:
            metrics = {}
            for line in f:
                key, value = line.strip().split(": ")
                metrics[key] = float(value)

        # 存储数据
        test_routes[test_num].add(int(route_num))
        test_time_metrics[test_num][time_key].update(metrics.keys())
        for metric, value in metrics.items():
            data[test_num][time_key][metric][route_key] = value

    # 为每个test生成CSV文件
    for test_num in data:
        # 准备排序参数
        routes = [f"route{i}" for i in sorted(test_routes[test_num])]
        time_order = sorted(
            data[test_num].keys(),
            key=lambda x: int(x.replace("min", "")),
            reverse=False
        )
        metric_order = ['mse', 'mae', 'aed', 'smape', 'fde']

        # 构建CSV内容
        rows = []
        for time in time_order:
            for metric in metric_order:
                if metric not in data[test_num][time]:
                    continue
                row = [f"{time} {metric}"]
                for route in routes:
                    value = data[test_num][time][metric].get(route, 0)
                    row.append(f"{value:.9f}".rstrip('0').rstrip('.') if value else '0')
                rows.append(row)

        # 写入文件
        csv_name = f"gru_{test_num}.csv"
        csv_path = os.path.join(path_a, csv_name)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头（route1到routeN）
            header = ["Time_Metric"] + routes
            writer.writerow(header)
            # 写入数据行
            writer.writerows(rows)
        print(f"生成文件：{csv_path}")
