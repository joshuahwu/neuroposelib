HEURISTICS_DICT = {
    "mouse20_notail": {
        'Bad_Tracking': {
            'high': [
                'abs_vel_vec_Snout_3',
                'abs_vel_vec_Tail_base__3',
                'abs_vel_vec_SpineM_3',
                'abs_vel_std_vec_Snout_3',
                'abs_vel_std_vec_Tail_base__3',
                'abs_vel_std_vec_SpineM_3',
            ],
            'low': []
        },
        'Rear': {
            'high': [
                'rel_vel_z_Snout_31',
                'rel_vel_z_EarR_31',
                'rel_vel_z_EarL_31',
                'abs_vel_z_Snout_31',
                'abs_vel_z_Snout_89',
                'abs_vel_z_SpineF_31',
                'abs_vel_z_SpineF_89',
                'ego_euc_Snout_z',
                'ego_euc_EarR_z',
                'ego_euc_EarL_z',
            ],
            'low': [
                'abs_vel_vec_Tail_base__31',
                'abs_vel_z_Tail_base__31',
                'abs_vel_vec_Tail_base__89',
                'abs_vel_z_Tail_base__89',
                'abs_vel_std_vec_Tail_base__3',
                'abs_vel_std_x_Tail_base__3',
                'abs_vel_std_y_Tail_base__3',
                'abs_vel_std_z_Tail_base__3',
                'abs_vel_std_vec_Tail_base__31',
                'abs_vel_std_x_Tail_base__31',
                'abs_vel_std_y_Tail_base__31',
                'abs_vel_std_z_Tail_base__31',
                'abs_vel_std_vec_Tail_base__89',
                'abs_vel_std_x_Tail_base__89',
                'abs_vel_std_y_Tail_base__89',
                'abs_vel_std_z_Tail_base__89',
            ]
        },
    }
}