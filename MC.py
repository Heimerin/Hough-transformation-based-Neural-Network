import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 


#warstwy detektora w milimetrach, przykladowe
LAYER_RAD = np.array([30,60,90,120,150,180,210,240])
#stale do modyfikacji
MAGNETIC_CONST = 0.0015
DETECTOR_RES=0.002 #rozmycie katowe w radianach

def generate_event(true_tracks = 10, noise_hits = 50, q_pt_range = (-2.0, 2.0)):
  hits = [] #do zapisu wspolrzednych x y z z warstwy oraz numer trajektorii
  true_params = [] #do zapisu prawdziwych wartosci (q/pT, phi) dla kazdej wygenerowanej cząstki

  phi_initial=np.random.uniform(0, 2*np.pi, true_tracks)
  q_pts = np.random.uniform(q_pt_range[0], q_pt_range[1], true_tracks)

  for id in range(true_tracks):

    q_pt = q_pts[id]
    phi = phi_initial[id]
    true_params.append([q_pt, phi])

    eps=1e-9
    if abs(q_pt) <eps:
      q_pt = eps*np.sign(q_pt) if q_pt != 0 else eps

    R = 1.0 / (2.0 * MAGNETIC_CONST * q_pt)

    for layer_id, layer_r in enumerate(LAYER_RAD):
      arg_arcsin = layer_r / (2*R)
      if abs(arg_arcsin) <= 1:
        ideal_phi = phi + np.arcsin(arg_arcsin)

        phi_hit_mod = ideal_phi + np.random.normal(0, DETECTOR_RES)

        x_hit = layer_r * np.cos(phi_hit_mod)
        y_hit = layer_r * np.sin(phi_hit_mod)
        hits.append([x_hit, y_hit, layer_r, id])
      else:
        break

  #generowanei szumu
  for i in range(noise_hits):
      random_layer = np.random.choice(LAYER_RAD)
      random_phi = np.random.uniform(0,2 * np.pi)
      x_noise = random_layer * np.cos(random_phi)
      y_noise= random_layer*np.sin(random_phi)

      hits.append([x_noise, y_noise, random_layer, -1]) #do etykiet id = -1
      #dodatkowe przetasowanie zey potem model nie uczyl sie jakiejs kolejnsoci 
  hits_arr = np.array(hits)
  np.random.shuffle(hits_arr)

  return hits_arr, np.array(true_params)
    

def event_visualize(hits_arr, true_params):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(111, projection = 'polar')
  for layer in LAYER_RAD:
    theta_circ = np.linspace(0, 2 * np.pi, 360)
    ax.plot(theta_circ, np.full_like(theta_circ, layer), color = 'gray', linestyle = '--', linewidth = 0.5)
  
  n_true_tracks = len(true_params)
  colors = cm.rainbow(np.linspace(0,1, n_true_tracks))
  for t_id in range(n_true_tracks):
    q_pt, phi_0 = true_params[t_id]
    eps=1e-9
    if abs(q_pt) <eps:
      q_pt = eps*np.sign(q_pt) if q_pt != 0 else eps

    R = 1.0 / (2.0 * MAGNETIC_CONST * q_pt)
    max_r = min(2.0 * abs(R), LAYER_RAD[-1])
    r_points = np.linspace(0.1, max_r, 200)

    ar_arcsin = r_points / (2*R)
    phi_points = phi_0 + np.arcsin(ar_arcsin)
    ax.plot(phi_points, r_points, color = colors[t_id], label = f'Track {t_id}')
  
  for hit in hits_arr:
    x_hit, y_hit, j, t_id = hit
    theta_hit = np.atan2(y_hit, x_hit) 
    if theta_hit < 0:
      theta_hit += 2.0 * np.pi
    r_hit = np.sqrt(x_hit**2 + y_hit**2)
    if t_id == -1:
      ax.scatter(theta_hit, r_hit, color = 'black', marker='.', s=10)
    else:
      ax.scatter(theta_hit, r_hit, color=colors[int(t_id)], marker='x', s=45)
    
  
  ax.set_theta_zero_location('N')
  ax.set_theta_direction(-1)
  ax.set_rmax(LAYER_RAD[-1] + 20)
  ax.set_yticklabels([])
  plt.title('MC simulation with noise hits')
  plt.show()

    
hit_package, particle_params = generate_event(true_tracks = 5, noise_hits=20)
# print(generate_event(true_tracks = 5, noise_hits=20)) 
event_visualize(hit_package, particle_params)

    


