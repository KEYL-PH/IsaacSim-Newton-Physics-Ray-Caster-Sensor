import numpy as np

def setup(db: og.Database):
    pass

def cleanup(db: og.Database):
    pass

def compute(db: og.Database):
    state = db.per_instance_state
    manual_control = db.inputs.manual_control

    state.forward = 0
    state.side = 0
    state.rotate = 0

    if manual_control:
        state.forward = -db.inputs.forward
        state.side = db.inputs.side
        state. rotate = db.inputs.rotate

#Offset Values "Multipliers"
    state.multiplier = 5
    state.rotate = state.rotate * state.multiplier
    state.side = state.side * state.multiplier

#Normalize side and rotate values
    state.angular_vel_f = state.rotate - state.side
    state.angular_vel_r = state.rotate + state.side
    state.vector = [state.angular_vel_f, state.angular_vel_r]

    if any(x>state.multiplier or x<-state.multiplier for x in state.vector):
        state.vector = (state.vector/np.max(np.abs(state.vector))) * state.multiplier

    db.outputs.front_linear_velocity = state.forward
    db.outputs.rear_linear_velocity = state.forward
    db.outputs.front_angular_velocity = state.vector[0]
    db.outputs.rear_angular_velocity = state.vector[1]
    return True