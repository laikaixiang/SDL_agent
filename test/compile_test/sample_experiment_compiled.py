# 自动生成的实验执行代码
import time

# 用户输入变量存储
user_vars = {}

def execute_experiment():
    # 用户输入
    user_vars['loop_count'] = input('请输入循环次数: ')
    # 循环执行3次
    for _loop_iter in range(3):
        # 用户输入目标温度
        user_vars['target_temp'] = input('请输入目标温度: ')
        # 判断温度是否大于100
        if int(user_vars.get('target_temp', 0)) > 100:
            # 设置高温
            print('执行硬件操作: set_temperature')
            # TODO: 调用硬件函数 set_temperature({'temperature': 150})
            # 等待2秒
            time.sleep(2.0)  # 等待 2.0 秒
        # 等待1秒
        time.sleep(1.0)  # 等待 1.0 秒
    # 清理步骤组
    # GROUP: 清理步骤
    for _group_iter in range(1):
        # 恢复室温
        print('执行硬件操作: set_temperature')
        # TODO: 调用硬件函数 set_temperature({'temperature': 25})

if __name__ == '__main__':
    execute_experiment()
