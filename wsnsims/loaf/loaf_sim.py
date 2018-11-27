import logging

import numpy as np
import math
import matplotlib.path as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

from wsnsims.core import cluster
from wsnsims.core import segment
from wsnsims.core.environment import Environment
from wsnsims.core.data import segment_volume
from wsnsims.core import point
from wsnsims.loaf import loaf_runner

logger = logging.getLogger(__name__)


class LOAF(object):
    def __init__(self, environment):
        """

        :param environment:
        :type environment: core.environment.Environment
        """
        self.env = environment
        locs = np.random.rand(self.env.segment_count, 2) * self.env.grid_height
        self.segments = [segment.Segment(nd) for nd in locs]

        for i, seg in enumerate(self.segments):
            seg.segment_id = i

        self.clusters = []

        self._cluster_center = None
        self._center = None

        self._plot = True

    def build_cluster(self, segment_ids, relay):
        new_cluster = cluster.BaseCluster(self.env)

        for seg in segment_ids:
            new_cluster.add(self.segments[seg])

        new_cluster.relay_node = self.segments[relay]
        return new_cluster

    def calculate_data_volume(self, given_segment):
        """
        For a given segment, calculate the data volume (Data(S_i))
        :param given_segment: the segment whose data volume is computed
        :return: the sum of the data volume between the given segment and each other segments
        """
        current_data_volume = 0
        for other_segment in self.segments:
            if given_segment is other_segment:
                # Do not calculate the data exchange between the node and the node again
                continue
            # Use the existing 'segment_volume' function to calculate the data exchanged
            current_data_volume += segment_volume(given_segment, other_segment, self.env)
        return current_data_volume

    def compute_center(self) -> point.Vec2:
        """
        Compute the center of mass eG
        :return: the center of mass as a point (not a segment)
        """
        c_x = 0
        c_y = 0
        data_all_segments = 0
        for given_segment in self.segments:
            current_data_segment = self.calculate_data_volume(given_segment)
            c_x += given_segment.location.x * current_data_segment
            c_y += given_segment.location.y * current_data_segment
            data_all_segments += current_data_segment

        c_x /= data_all_segments
        c_y /= data_all_segments
        center = point.Vec2([c_x, c_y])

        return center

    def get_closest(self, location: point.Vec2) -> segment.Segment:
        """
        Return the segment that is the closest to the given point
        :param location: the vector of the coordinate of the given point
        :return: the segment that is the closest to the given point
        """
        # Start by getting an arbitrary minimum
        minimum_distance = location.distance(self.segments[0].location)
        closest_segment = self.segments[0]

        # For all other segments, check if the distance to the location is smaller
        for seg in self.segments[1:]:
            current_distance = location.distance(seg.location)
            if current_distance < minimum_distance:
                minimum_distance = current_distance
                closest_segment = seg
        return closest_segment

    def _minimize_energy(self, given_cluster: list) -> tuple:
        """
        Given a list of clusters, get the two clusters, which, when merged, reduce the most the overall energy.
        :param given_cluster: the list of clusters
        :return: a tuple of the index of the two clusters
        """
        energy_minimum = -1
        index_chosen_cx = None
        index_chosen_cy = None
        number_remaining_segments = len(given_cluster)

        for x in range(number_remaining_segments):
            for y in range(number_remaining_segments):
                # Preventing selecting two times the same cluster
                if x == y:
                    continue
                energy_sum = 0

                # Sum of energy for i=1 to N(O)cluster - r, i =/= x,y
                for i in range(number_remaining_segments):
                    # The C_i is supposed different from C_x and C_y
                    if i == x or i == y:
                        continue

                    ci_id = given_cluster[i].cluster_id

                    merge_cx_cy = given_cluster[x].merge(given_cluster[y])

                    # Build a temporary simulation with the Cx and Cy removed, (Cx U Cy) added
                    temp_sim = LOAF(self.env)
                    temp_sim.clusters = given_cluster.copy()
                    temp_sim.clusters.append(merge_cx_cy)
                    temp_sim.clusters.remove(given_cluster[x])
                    temp_sim.clusters.remove(given_cluster[y])

                    runner = loaf_runner.LOAFRunner(temp_sim, temp_sim.env)

                    # energy_sum += e(Ci) + me(Cx U Cy)
                    energy_sum += runner.energy_model.total_energy(ci_id)
                    energy_sum += runner.energy_model.total_energy(merge_cx_cy.cluster_id)

                if energy_sum < energy_minimum or energy_minimum == -1:  # "-1" condition: to have a first minimum
                    energy_minimum = energy_sum
                    index_chosen_cx = x
                    index_chosen_cy = y

        return index_chosen_cx, index_chosen_cy

    def first_phase(self):
        # 3.
        center_location = self.compute_center()
        center = self.get_closest(center_location)
        self._center = center

        # 4. Ck = {Si | EuclideanDist(Si , eG) ≤ R };
        segment_ids = []
        for s in self.segments:
            # print("segment {}: {}".format(s.segment_id, s.location.distance(center.location)))
            if s.location.distance(center.location) <= self.env.comms_range:
                segment_ids.append(s.segment_id)

        original_cluster = self.build_cluster(segment_ids, center.segment_id)
        self.clusters.append(original_cluster)

        if self.env.mdc_count == 1:
            raise NotImplementedError("What to do if mdc_count equals 1?")

        # 5. S −= C k
        remaining_segments = [seg for seg in self.segments if seg not in original_cluster.nodes]

        # 6. j = 0
        # 7. for each Si ∈ (S − Ck){
        # 8.  Cj = { Si , eG }; j++;
        # 9. } end for
        noncentral_clusters = []
        for seg in remaining_segments:
            segment_list = [seg.segment_id, center.segment_id]
            noncentral_clusters.append(self.build_cluster(segment_list, center.segment_id))

        # 10. r = 1; N(0)cluster = |S| − |Ck|; Ncluster = N(0)cluster;
        round_num = 1
        number_remaining_segments = len(remaining_segments)
        index_n_cluster = number_remaining_segments

        # 16. while (r < N cluster−(k − 1))
        k = self.env.mdc_count
        while True:

            # 11-12. Cx , Cy = min(x, y, (∑ i=1,i≠x,y E(C i ) + ME(C x ∪ C y ))
            # 13. where ME(Cx, Cy ) = EM (Cx ∪ Cy ) + EC (Cx ∪ Cy )
            index_cluster_x, index_cluster_y = self._minimize_energy(noncentral_clusters)

            # 14 Cx U= Cy ;
            cluster_x = noncentral_clusters[index_cluster_x]
            cluster_y = noncentral_clusters[index_cluster_y]
            merged_cluster = cluster_x.merge(cluster_y)
            merged_cluster.relay_node = center
            noncentral_clusters[index_cluster_x] = merged_cluster

            # Cy = CN cluster ;
            # TODO: not sure here: so I just removed the C_y cluster from the list, as it was mergec into C_x
            # noncentral_clusters[index_cluster_y] = noncentral_clusters[index_n_cluster]
            noncentral_clusters.pop(index_cluster_y)

            # 15. r = r + 1
            round_num += 1
            # Ncluster -= 1
            index_n_cluster -= 1

            self.clusters = noncentral_clusters.copy()
            # self.plotting()

            # 16. while (r < N cluster−(k − 1)) (to emulate a do-while)
            if not round_num < (number_remaining_segments - (k - 1)):
                break

        self.clusters = noncentral_clusters.copy()
        # add Ck in k-th position
        self.clusters.append(original_cluster)
        # Just re-label the clusters for display
        for i, clust in enumerate(self.clusters):
            clust._cluster_id = i

        self.plotting()
        return center

    @staticmethod
    def _calc_rdv_point(center: point.Vec2, center_of_mass: point.Vec2, distance_from_center: float,
                        cluster_id: int) -> segment.Segment:
        """
        Create a new Segment on the line between center and center_of_mass, with the distance "distance_from_center"
        from the "center" point
        :param center:
        :param center_of_mass:
        :param distance_from_center:
        :param cluster_id: the ID of the cluster to which the Segment will be added
        :return: the new Segment
        """
        dst = center_of_mass.distance(center)
        angle = math.acos((center_of_mass.x - center.x) / dst)
        angle2 = math.asin((center_of_mass.y - center.y) / dst)

        # Need this because the acos gives only result in [0, pi] (so acos(pi/2) == acos(-pi/2))
        if angle2 < 0:
            angle *= -1

        # Basic trigonometry
        x_rdv_point = distance_from_center * math.cos(angle) + center.x
        y_rdv_point = distance_from_center * math.sin(angle) + center.y

        rendezvous_point = segment.Segment([x_rdv_point, y_rdv_point])
        rendezvous_point.fake_segment = cluster_id  # Attribute to express it is a virtual segment, not a real node

        assert round(dst * math.cos(angle) + center.x) == round(center_of_mass.x)
        assert round(dst * math.sin(angle) + center.y) == round(center_of_mass.y)

        # direction_x = int(round(abs(center.x - center_of_mass.x) / (center.x - center_of_mass.x)))
        # direction_y = int(round(abs(center.y - center_of_mass.y) / (center.y - center_of_mass.y)))

        # if int(round(rendezvous_point.location.x)) in range(int(round(center_of_mass.x)), int(round(center.x)),
        #                                                     direction_x) and\
        #         int(round(rendezvous_point.location.y)) in range(int(round(center_of_mass.y)),
        #                                                          int(round(center.y)), direction_y):
        #     assert distance_from_center <= dst
        # else:
        #     assert distance_from_center > dst

        assert round(distance_from_center) == round(center.distance(rendezvous_point.location))

        # print("Center: {} to CoM: {}, rdv: {}".format(center.nd, center_of_mass.nd, rendezvous_point.location.nd))

        return rendezvous_point

    @staticmethod
    def _create_dict_length_list(orig_dict: dict, new_key: object, length: int) -> dict:
        """
        In a dictionary, create for the given key a list of None elements with the given size
        :param orig_dict: the dictionary to which add an list
        :param new_key: the key to match the new list
        :param length: the size of the new list of None
        :return: the dictionary given, but with the new key and value
        """
        if new_key not in orig_dict:
            orig_dict[new_key] = [None] * length  # Create a list of "None" of size length
        return orig_dict

    def _replace_rdv_point(self, given_cluster: cluster, rdv_point: segment.Segment) -> None:
        """
        In a chosen cluster, replace the older rendezvous point (Pi) by a new one
        :param given_cluster: the cluster in which the point has to be replaced
        :param rdv_point: the new rendezvous point
        :return:
        """
        # Get the cluster to which the rendezvous point belong (useful to replace it in the center cluster)
        if hasattr(rdv_point, "fake_segment"):
            bound_cluster_id = rdv_point.fake_segment
        else:
            raise ValueError("The point given as rendezvous point given is not.")

        # Get the old rendezvous point
        seg_to_remove = None
        for seg in given_cluster.nodes:
            if hasattr(seg, "fake_segment") and seg.fake_segment == bound_cluster_id:
                seg_to_remove = seg

        # Remove the old rendezvous point from the current cluster, the central cluster and the list of segments
        if seg_to_remove:
            given_cluster.remove(seg_to_remove)
            self._cluster_center.remove(seg_to_remove)
            self.segments.remove(seg_to_remove)

        # Add the new rendezvous point in the central cluster and as the relay node of the cluster.
        given_cluster.add(rdv_point)
        given_cluster.relay_node = rdv_point
        self._cluster_center.add(rdv_point)

        # If the new rendezvous points was not in the list of Segments, add it
        if rdv_point not in self.segments:
            self.segments.append(rdv_point)

    @staticmethod
    def _calc_center_of_mass(clust: cluster) -> point.Vec2:
        """
        For a given cluster, calculate its center of mass
        :param clust:
        :return: the center of mass as a coordinate vector
        """
        hull_verts = clust.tour.hull
        # The center is the center of mass of the polygon built by the tour of the cluster
        points = clust.tour.points
        return point.Vec2(np.average([node for node in points[hull_verts]], axis=0))

    def plotting(self, round=None):
        """
        If the '_plot' attribute is set to True, plot the current state of the clusters, with the Segments and paths.
        :return:
        """
        if not self._plot:
            return

        x_coor = []
        y_coor = []

        x_rdv = []
        y_rdv = []

        names = []
        x_whole = []
        y_whole = []
        for seg in self.segments:
            names.append(seg.segment_id)
            # Separate the P_i from the other points
            if hasattr(seg, "fake_segment"):
                x_rdv.append(seg.location.x)
                y_rdv.append(seg.location.y)
            else:
                x_coor.append(seg.location.x)
                y_coor.append(seg.location.y)
            x_whole.append(seg.location.x)
            y_whole.append(seg.location.y)

        x_center, y_center = self._center.location.nd

        fig, ax = plt.subplots()

        colors = ["black", "red"]

        # Print the tour path of each cluster
        for clust in self.clusters:
            points = clust.tour.points
            hull_verts = clust.tour.hull
            clust_path = mp.Path(points[hull_verts], closed=True)

            last_clust = int(clust is self.clusters[-1])  # If the cluster is the central cluster, display it red
            patch = patches.PathPatch(clust_path, facecolor='none', lw=2, edgecolor=colors[last_clust])
            ax.add_patch(patch)

        ax.plot(x_coor, y_coor, 'bs')
        ax.plot(x_rdv, y_rdv, 'g^')
        ax.plot(x_center, y_center, 'ro')
        for i, txt in enumerate(names):
            ax.annotate(txt, (x_whole[i], y_whole[i]))
        if round:
            fig.suptitle("State at round {}".format(round))
        plt.show()

    def second_phase(self):

        # Important: The last cluster in self.clusters is the central cluster
        c_k = self.clusters[-1]
        self._cluster_center = c_k
        epsilon = self.env.comms_range

        # 1. for ∀i { // based on the initial inter-cluster topology formed during the 1st phase
        round_num = 0
        tour_paths = self._create_dict_length_list({}, round_num, len(self.clusters))
        rendezvous_points = self._create_dict_length_list({}, round_num, len(self.clusters))
        mdc_energies = self._create_dict_length_list({}, round_num, len(self.clusters))
        centers_of_mass = self._create_dict_length_list({}, round_num, len(self.clusters))
        list_values = [tour_paths, rendezvous_points, mdc_energies, centers_of_mass]

        runner = loaf_runner.LOAFRunner(self, self.env)

        centers_of_mass[round_num][-1] = self._calc_center_of_mass(self.clusters[-1])

        for index, clust in enumerate(self.clusters[:-1]):

            # TODO: everything is bad here: TR depends on Pi, which depends on CoMi which depends on TR
            # 2. C i0 <- C i , in the 1 st phase;
            # The non-central clusters are all clusters in self.clusters except the last one

            # 3. TR i0 <- Tour path of M i in C i computed by [25];
            tour_paths[round_num][index] = clust.tour

            # 5. E i0 <- Energy consumed by M i in C i ;
            mdc_energies[round_num][index] = runner.energy_model.total_energy(clust.cluster_id)

            # 6. CoM i0 <- Core of mass of a polygon formed by TR i0 ;
            com_i = self._calc_center_of_mass(clust)
            centers_of_mass[round_num][index] = com_i

            # 4. P i0 <- Rendezvous point where M i meets M k ;
            rendezvous_points[round_num][index] = self._calc_rdv_point(centers_of_mass[round_num][-1], com_i,
                                                                       self.env.comms_range, index)

            # 7. } end for
            # clust._invalidate_cache()

        centers_of_mass[round_num][-1] = self._calc_center_of_mass(self.clusters[-1])

        self.plotting(round=round_num)

        mdc_energies[round_num][-1] = runner.energy_model.total_energy(self.clusters[-1].cluster_id)
        del runner

        # 8. E AVG <- Average(E i0 , ∀i); // E ko < E AVG
        e_averages = [np.average(mdc_energies[round_num])]
        # 9. SD <- Standard Deviation of E i ∀i; rr <- 0;
        old_sd = np.std(mdc_energies[round_num])
        sds = [old_sd]
        rr = 0

        # 10. do {
        while True:
            # 11. r = 1; // 1 st step during which P m ’s move outward
            round_num = 1

            self.plotting(round=round_num)

            # Prepare the lists for the new round
            for index, values in enumerate(list(list_values)):
                list_values[index] = self._create_dict_length_list(values, round_num, len(self.clusters))

            # 12. do {
            while True:
                # print("In first while, r: {}".format(round_num))

                centers_of_mass[round_num][-1] = self._calc_center_of_mass(self.clusters[-1])

                for index, values in enumerate(list(list_values)):
                    list_values[index] = self._create_dict_length_list(values, round_num, len(self.clusters))

                # 13. E least = min E ir−1 ;
                e_least = min(mdc_energies[round_num - 1][:-1])

                # 14. for E ir−1 ≥ E AVG , ∀i {
                # For each cluster, move the Pi outwards when the energy of the cluster is more than average
                for index, mdc_energy in enumerate(mdc_energies[round_num - 1][:-1]):
                    old_rdv_point = rendezvous_points[round_num - 1][index]
                    if not mdc_energy >= e_averages[round_num - 1]:
                        # If the energy is less than average, choose the new rendezvous point as the old one
                        rendezvous_points[round_num][index] = old_rdv_point
                        continue

                    # 15. L r i = Line from P ir−1 towards CoM ir−1 ;
                    # 16. P ir moves (ε × /) meters along L r i ;
                    shifting_ratio = epsilon * mdc_energies[round_num - 1][index] / e_least

                    # dist = centers_of_mass[round_num][-1].distance(old_rdv_point.location)
                    dist = centers_of_mass[round_num][-1].distance(old_rdv_point.location)

                    # Shift the Pi from its current distance (dst) with the shifting ratio (-> "dist + shifting_ratio")
                    new_rdv_point = self._calc_rdv_point(centers_of_mass[round_num][-1],
                                                         old_rdv_point.location,
                                                         dist + shifting_ratio, self.clusters[index].cluster_id)

                    assert dist <= new_rdv_point.location.distance(centers_of_mass[round_num][-1])

                    self.segments.append(new_rdv_point)
                    rendezvous_points[round_num][index] = new_rdv_point

                    # If the point eG is still present in the cluster, remove it
                    # -> to replace it with a rendezvous point
                    seg_to_remove = None
                    for seg in self.clusters[index].nodes:
                        if seg is self._center:
                            seg_to_remove = seg
                    if seg_to_remove:
                        self.clusters[index].remove(seg_to_remove)

                    self._replace_rdv_point(self.clusters[index], new_rdv_point)

                    # print("Moved segment: {} in cluster {}".format(new_rdv_point.segment_id,
                    #                                                new_rdv_point.cluster_id))

                    # 17. } end for

                self.plotting(round=round_num)

                # 18. Update TR kr based on P ir ;
                tour_paths[round_num][-1] = c_k.tour

                # 19. if ∃S x ∈ C ir in a polygon formed by TR kr then {
                points = c_k.tour.points
                hull_verts = c_k.tour.hull
                ck_path = mp.Path(points[hull_verts])
                for clust in self.clusters[:-1]:
                    for seg in list(clust.nodes):  # Has to copy the list to be able to remove from it

                        # If the segment is not a rendezvous point and it is in the polygon formed by Tr
                        if not hasattr(seg, "fake_segment") and ck_path.contains_point(seg.location.nd):
                            # 20. C kr ∪= {S x }; C ir −= {S x };
                            # Then add this segment to C_k
                            clust.remove(seg)
                            c_k.add(seg)

                # 21. } end if

                # 22. Compute TR ir and CoM ir , i<k and E ir , ∀i;

                # TR ir, CoM ir, i<k
                for index, clust in enumerate(self.clusters[:-1]):
                    # Compute TR ir
                    new_rdv_point = rendezvous_points[round_num][index]
                    if new_rdv_point is not None:

                        self._replace_rdv_point(clust, new_rdv_point)

                    tour_paths[round_num][index] = clust.tour

                    # Compute CoM ir
                    com_i = self._calc_center_of_mass(clust)
                    centers_of_mass[round_num][index] = com_i

                # E(r)_i for all i
                runner = loaf_runner.LOAFRunner(self, self.env)
                for index, clust in enumerate(self.clusters):
                    mdc_energies[round_num][index] = runner.energy_model.total_energy(clust.cluster_id)

                del runner

                # 23. E AVG = Average(E ir , ∀i); r += 1;
                for index, rdv_point in enumerate(rendezvous_points[round_num]):
                    if rdv_point is None:
                        rendezvous_points[round_num] = rendezvous_points[round_num - 1]

                e_averages.append(np.average(mdc_energies[round_num]))
                round_num += 1

                # 24. } while ( E kr < E AVG )
                # "- 1" added after round_num -->
                # print("mdc_energies[round_num - 1][-1]: {} e_averages[round_num - 1]: {}".format(
                #     mdc_energies[round_num - 1][-1], e_averages[round_num - 1]))
                # TODO, not sure: round_num is increased, but the energy cannot be calculated --> take "round_num - 1"
                if not mdc_energies[round_num - 1][-1] < e_averages[round_num - 1]:
                    break

            # 25. f = r = r − 1; // 2 nd step during which P l ’s move inward
            round_num -= 1
            f = round_num

            # Move the Pi inward eG
            number_tries = 10
            # 26. do {
            while True:
                # print("In second while, r: {}".format(round_num))

                for index, values in enumerate(list(list_values)):
                    list_values[index] = self._create_dict_length_list(values, f, len(self.clusters))

                # 27. CoM kr = CoM kf ;
                centers_of_mass[round_num][-1] = centers_of_mass[f][-1]

                # 28. for each C i whose E lr < E AVG {
                for index, mdc_energy in enumerate(list(mdc_energies[round_num][:-1])):  # Need a copy to update it
                    if not mdc_energy < e_averages[round_num]:
                        rendezvous_points[round_num][index] = rendezvous_points[round_num - 1][index]
                        continue
                    # 29. L i = Line from P i towards CoM kr ;
                    shifting_ratio = epsilon * e_least / mdc_energies[round_num - 1][index]

                    # 30. P ir moves (ε × /) meters along L r i ;
                    old_rdv_point = rendezvous_points[round_num - 1][index].location
                    dist = centers_of_mass[round_num][-1].distance(old_rdv_point)

                    # Prevent moving the Pi on the other side of eG
                    if dist <= shifting_ratio:
                        continue
                    # Move the Pi inward -> "dist - shifting_ratio"
                    new_rdv_point = self._calc_rdv_point(centers_of_mass[round_num][-1], old_rdv_point,
                                                         dist - shifting_ratio, self.clusters[index].cluster_id)

                    assert dist >= new_rdv_point.location.distance(centers_of_mass[round_num][-1])

                    self.segments.append(new_rdv_point)
                    rendezvous_points[round_num][index] = new_rdv_point

                    # 31. Update TR ir , E ir , and CoM ir ;
                    self._replace_rdv_point(self.clusters[index], new_rdv_point)
                    runner = loaf_runner.LOAFRunner(self, self.env)

                    tour_paths[round_num][index] = self.clusters[index].tour
                    mdc_energies[round_num][index] = runner.energy_model.total_energy(self.clusters[index].cluster_id)
                    new_coi = self._calc_center_of_mass(self.clusters[index])
                    centers_of_mass[round_num][index] = new_coi
                    del runner
                # 32. } end for

                centers_of_mass[round_num][-1] = self._calc_center_of_mass(self.clusters[-1])

                self.plotting(round=round_num)

                # 33. r += 1;
                for index, rdv_point in enumerate(rendezvous_points[round_num]):
                    if rdv_point is None:
                        rendezvous_points[round_num] = rendezvous_points[round_num - 1]

                # Calculate the clusters for the round r-1, to be able to return it
                temp_list = []
                for clust in self.clusters:
                    temp_list.append(copy.deepcopy(clust))
                cluster_r_minus_one = temp_list
                round_num += 1

                for index, values in enumerate(list(list_values)):
                    list_values[index] = self._create_dict_length_list(values, round_num, len(self.clusters))

                # 34. } while ( E kr > E AVG )
                runner = loaf_runner.LOAFRunner(self, self.env)
                for index, clust in enumerate(self.clusters):
                    mdc_energies[round_num][index] = runner.energy_model.total_energy(clust.cluster_id)
                e_averages.append(np.average(mdc_energies[round_num]))

                del runner

                print("E(r)_k: {} E(r)_AVG: {}".format(
                      mdc_energies[round_num][-1], e_averages[round_num]))

                number_tries -= 1
                if number_tries <= 0:
                    break

                if not mdc_energies[round_num][-1] > e_averages[round_num]:
                    break

            # 35. rr += 1; SD rr <- Standard Deviation of E ir ∀i;
            sds.append(np.std(mdc_energies[round_num]))
            rr += 1

            # 36. } while ( SD rr < SD rr−1 )
            if not sds[rr] < sds[rr - 1]:
                break

        # 37. return { C ir−1 , TR ir−1 ∀i }
        return cluster_r_minus_one, tour_paths[round_num - 1]

    def run(self):
        """

        :return:
        :rtype: loaf.loaf_runner.LOAFRunner
        """
        self.first_phase()

        self.second_phase()

        runner = loaf_runner.LOAFRunner(self, self.env)

        logger.debug("Maximum comms delay: {}".format(
            runner.maximum_communication_delay()))
        logger.debug("Energy balance: {}".format(runner.energy_balance()))
        logger.debug("Average energy: {}".format(runner.average_energy()))
        logger.debug("Max buffer size: {}".format(runner.max_buffer_size()))
        return runner


def main():
    env = Environment()
    # seed = int(time.time())

    # General testing ...
    # seed = 1484764250
    # env.segment_count = 12
    # env.mdc_count = 5

    seed = 1487736569
    env.comms_range = 125

    # Specific testing ...
    # seed = 1483676009  # center has in-degree of 3
    # seed = 1483998718  # center has in-degree of 2

    logger.debug("Random seed is %s", seed)
    np.random.seed(seed)
    sim = LOAF(env)
    sim.run()
    sim.show_state()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('loaf_sim')
    main()
