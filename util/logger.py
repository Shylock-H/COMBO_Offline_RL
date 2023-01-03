import os
import atexit

class Logger:
    def __init__(self, writer, log_name = 'progess.txt') -> None:
        self.writer = writer
        self.log_file = open(os.path.join(self.writer.get_logdir(), log_name), 'w')
        atexit.register(self.log_file.close)
    
    def record(self, tag, value, step, display = True):
        self.writer.add_scalar(tag, value, step)
        if display:
            info = f"{tag}: {value:.3f}"
            self.print(info)
    
    def print(self, info):
        print('\033[1;32m [info]\033[0m : ' + info)
        self.log_file.write(info + '\n')
