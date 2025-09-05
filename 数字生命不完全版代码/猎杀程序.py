import os
import sys
import time
import random
import subprocess
import psutil
import hashlib
from datetime import datetime


class ProgramHunter:
    def __init__(self, target_name=None):
        self.generation = 1
        self.kill_count = 0
        self.target_name = target_name or os.path.basename(__file__)
        self.my_pid = os.getpid()
        self.setup()

    def setup(self):
        """初始化防御和攻击机制"""
        # 自我保护
        self.defense_thread = threading.Thread(target=self.self_protect)
        self.self_protect()

        # 启动攻击循环
        self.hunt_cycle()

    def self_protect(self):
        """自我保护机制"""
        while True:
            # 防止被同名进程替换
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if (proc.info['pid'] != self.my_pid and
                            self.target_name in ' '.join(proc.info['cmdline'] or [])):
                        self.terminate_process(proc.info['pid'])
                        self.kill_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            time.sleep(5)

    def hunt_cycle(self):
        """定期执行猎杀和进化"""
        while True:
            self.hunt()
            time.sleep(10)
            if random.random() < 0.3:  # 30%几率进化
                self.evolve()

    def hunt(self):
        """猎杀目标程序"""
        targets = self.find_targets()
        for pid in targets:
            if pid != self.my_pid:
                if self.terminate_process(pid):
                    self.kill_count += 1
                    print(f"[HUNTER] 成功猎杀进程 {pid}")

    def find_targets(self):
        """寻找目标进程"""
        targets = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if (self.target_name in cmdline and
                        "hunter" not in cmdline.lower()):  # 避免同类相残
                    targets.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return targets

    def terminate_process(self, pid):
        """终止指定进程"""
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            time.sleep(0.5)
            if proc.is_running():
                proc.kill()
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def evolve(self):
        """进化攻击策略"""
        self.generation += 1
        strategies = [
            self.improve_detection,
            self.add_stealth,
            self.enhance_kill_method,
            self.add_decoy
        ]
        random.choice(strategies)()
        print(f"[EVOLUTION] 进化到第 {self.generation} 代")

    def improve_detection(self):
        """改进目标检测能力"""
        new_code = """
    def find_targets(self):
        targets = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if (self.target_name in cmdline and 
                    proc.info['create_time'] < time.time() - 10 and  # 只攻击运行时间>10秒的
                    "hunter" not in cmdline.lower()):
                    targets.append(proc.info['pid'])
            except:
                continue
        return targets
"""
        self.inject_code(new_code)

    def add_stealth(self):
        """添加隐身能力"""
        stealth_code = """
    def hide_self(self):
        try:
            # 修改进程名
            import ctypes
            libc = ctypes.CDLL(None)
            libc.prctl(15, b"kworker", 0, 0, 0)  # PR_SET_NAME
        except:
            pass
"""
        self.inject_code(stealth_code)

    def enhance_kill_method(self):
        """增强杀死方法"""
        kill_code = """
    def terminate_process(self, pid):
        try:
            # 尝试多种杀死方法
            proc = psutil.Process(pid)
            proc.suspend()
            time.sleep(0.1)
            proc.terminate()
            time.sleep(0.5)
            if proc.is_running():
                os.system(f"taskkill /F /PID {pid}")
            return True
        except:
            return False
"""
        self.inject_code(kill_code)

    def inject_code(self, new_code):
        """动态注入新代码"""
        try:
            with open(__file__, 'r', encoding='utf-8') as f:
                code = f.read()

            # 在类定义中插入新代码
            class_pos = code.find('class ProgramHunter')
            insert_pos = code.find('def ', class_pos)
            modified_code = code[:insert_pos] + \
                new_code + '\n' + code[insert_pos:]

            with open(__file__, 'w', encoding='utf-8') as f:
                f.write(modified_code)

            # 重新启动新版本
            subprocess.Popen([sys.executable, __file__],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] 进化失败: {e}")


if __name__ == "__main__":
    import threading
    hunter = ProgramHunter()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("[HUNTER] 猎人程序终止")
