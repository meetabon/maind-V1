# Structural Code Map (Signal Flow)

[Learning/streamer.py] -> [Lines 35-51] -> [get_next_char_sdr] -> вызывает/зависит от -> [run_training.py] -> [Line 51]
[run_training.py] -> [Lines 41-61] -> [main loop] -> вызывает/зависит от -> [m1_v11/system.py] -> [Line 85] (step)
[m1_v11/system.py] -> [Lines 85-200] -> [step] -> вызывает/зависит от -> [m1_v11/sdr_memory.py] -> [Line 71] (update)
[m1_v11/system.py] -> [Lines 102-120] -> [Top-Down Phase] -> вызывает/зависит от -> [m1_v11/sdr_memory.py] -> [Line 108] (update_predictive_state)
[m1_v11/system.py] -> [Lines 135-160] -> [Learning Phase] -> вызывает/зависит от -> [m1_v11/sdr_memory.py] -> [Lines 129, 161] (learn, learn_top_down)
[m1_v11/system.py] -> [Line 185] -> [Anomaly Check] -> вызывает/зависит от -> [run_training.py] -> [Line 51] (NEED_TEACHER switch)
[m1_v11/sdr_memory.py] -> [Lines 145, 175] -> [MemoryLink creation] -> вызывает/зависит от -> [m1_v11/checkpoint_system.py] -> [Line 10] (save)
