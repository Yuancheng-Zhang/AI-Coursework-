goal = 274 ##179 ##274 ##111
punish = -100
dis_val = 0.99
start = 1
R = [0, 0, 0, 0, 0]

Iteration = range(0, 3)

for i in Iteration:
    
    R[4] = dis_val * max(( 0.8*goal + 0.1*punish + 0.1*punish ), ( 0.8*R[3] + 0.1*punish + 0.1*punish ))
    R[3] = dis_val * max(( 0.8*R[4] + 0.1*punish + 0.1*punish ), ( 0.8*R[2] + 0.1*punish + 0.1*punish ))
    R[2] = dis_val * max(( 0.8*R[3] + 0.1*punish + 0.1*punish ), ( 0.8*R[1] + 0.1*punish + 0.1*punish ))
    R[1] = dis_val * max(( 0.8*R[2] + 0.1*punish + 0.1*punish ), ( 0.8*R[0] + 0.1*punish + 0.1*punish ))
    R[0] = dis_val * max(( 0.8*R[1] + 0.1*punish + 0.1*punish ), ( 0.8*start + 0.1*punish + 0.1*punish ))
    
print(R)
