  for i in range(len(target_hist)-1):
                    draw_line(
                        viewer,
                        target_hist[i],
                        target_hist[i+1],
                        np.array([1, 1, 0, 1]),
                        width=3
                    )
                    
                    
target_hist = []


                target_hist.append(p_target_true.copy())
                if len(target_hist) > 200:
                    target_hist.pop(0)