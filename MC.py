import numpy as np

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
      np.random.shuffle(hits)

      return hits, np.array(true_params)
    
hit_package, particle_params = generate_event(true_tracks = 5, noise_hits=20)
print(generate_event(true_tracks = 5, noise_hits=20))

    


