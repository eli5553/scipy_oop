# same doc
import math
import simpy
import random
import numpy as np
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import font_manager

# for lognormal distribution


class Lognormal:
    def __init__(self, mean, stdev, random_seed=None):
        self.rand = np.random.default_rng(seed=random_seed)
        mu, sigma = self.normal_moments_from_lognormal(mean, stdev**2)
        self.mu = mu
        self.sigma = sigma

    # return mu and sigma of normal distribution
    # lognormal distribution with mean m and variance v
    def normal_moments_from_lognormal(self, m, v):
        phi = math.sqrt(v + m**2)
        mu = math.log(m**2/phi)
        sigma = math.sqrt(math.log(phi**2/m**2))
        return mu, sigma

    def sample(self):
        return self.rand.lognormal(self.mu, self.sigma)


# variables for calculating average waiting time
totalpatients = 0
totaltime = 0

# parameters


class p:
    # interarrival time
    inter = 5

    # resources for AE
    number_docs = 40
    number_docs_not_urgent = 2
    number_nurses = 15
    number_tests = 40
    number_med = 30

    # average time spent at each resource
    mean_nurse_triage = 10
    stdev_nurse_triage = 5
    mean_doc_consult = 30
    stdev_doc_consult = 10
    mean_doc_noturgent = 20
    stdev_doc_noturgent = 8
    mean_test = 60
    stdev_test = 10
    mean_med_collect = 15
    stdev_med_collect = 2
    mean_ip_wait = 30

    # queuing times
    wait_triage = []
    wait_doc = []
    wait_doc_not_urgent = []
    wait_test = []
    wait_test_doc = []
    wait_med = []
    total_time_in_hospital = []
    interarrival_time = []
    # duration for each run
    duration = 48000
    runs = 1


class simpy.resources.store.Store(env, capacity=1):
    def __init__(self):
        self.items = 'doc'
        self.capacity = 40
        self.inuse = 0

    def store_put(self):
        self.inuse -= 1

    def store_get(self):
        self.inuse += 1


# patients information
class IncomingPatients:
    def __init__(self, p_id) -> None:
        self.p_id = p_id
        self.time_in_system = 0

    def set_come(self):
        self.come = random.choices(['ambulance', 'walkin'], [0.12, 0.88])[0]

    def set_priority(self):
        # 1 is the highest priority while 5 is the least priority
        if self.come == 'ambulance':
            self.priority = random.choices(
                [1, 2, 3, 4, 5], [0.1, 0.2, 0.4, 0.285, 0.015])[0]
        else:
            self.priority = random.choices(
                [1, 2, 3, 4, 5], [0.1, 0.2, 0.4, 0.2, 0.1])[0]

    def set_triage_outcome(self):
        # decision tree
        if self.priority < 5:
            self.triage_outcome = 'AE'
        elif self.priority == 5:
            self.triage_outcome = 'not_urgent'


# model of the hospital, accident and emergency department
class HospitalAE:
    def __init__(self) -> None:
        self.env = simpy.Environment()
        self.patient_counter = 0
        # urgent patients get seen by the doctors and get tested first
        self.doc = simpy.resources.store.Store(
            self.env, capacity=p.number_docs)
        self.test = simpy.PriorityResource(self.env, capacity=p.number_tests)

        # the rest: FIFO
        self.nurse = simpy.Resource(self.env, capacity=p.number_nurses)
        self.doc_not_urgent = simpy.Resource(
            self.env, capacity=p.number_docs_not_urgent)
        self.med = simpy.Resource(self.env, capacity=p.number_med)

    def generate_patients(self):
        while True:
            # add patient
            self.patient_counter += 1

            # patient ID
            ae_p = IncomingPatients(self.patient_counter)

            self.env.process(self.attend_hospital(ae_p))

            # randomly sample the time takent for next patient to arrive
            sampled_interarrival = random.expovariate(1.0 / p.inter)
            p.interarrival_time.append(sampled_interarrival)
            yield self.env.timeout(sampled_interarrival)

    # patient's path through emergency department
    def attend_hospital(self, patient):
        # set priority
        patient.set_come()
        patient.set_priority()
        patient.set_triage_outcome()
        enter_system_time = self.env.now

        # if ambulance + AE, no need to wait for triage
        if (patient.come == 'ambulance' and patient.triage_outcome == 'AE'):
            p.wait_triage.append(0)
        else:
            # request a nurse
            with self.nurse.request() as req:
                # freeze until request can be met
                yield req
                triage_queue_end = self.env.now
                p.wait_triage.append(triage_queue_end - enter_system_time)

                # sample triage time from lognormal
                lognorm = Lognormal(mean=p.mean_nurse_triage,
                                    stdev=p.stdev_nurse_triage)
                sampled_triage_duration = lognorm.sample()

                yield self.env.timeout(sampled_triage_duration)

        if patient.triage_outcome == 'AE':
            doc_queue_start = self.env.now
            with simpy.resources.store.StoreGet(self.doc) as req_doc:
                yield req_doc
                doc_queue_end = self.env.now
                p.wait_doc.append(doc_queue_end - doc_queue_start)
                # sample consult time from lognormal
                lognorm = Lognormal(mean=p.mean_doc_consult,
                                    stdev=p.stdev_doc_consult)
                sampled_consult_duration = lognorm.sample()

                yield self.env.timeout(sampled_consult_duration)

            # 80% of patients needs a test
            test_prob = random.uniform(0, 1)
            if test_prob < 0.8:
                test_queue_start = self.env.now

                with self.test.request(priority=patient.priority) as req_test:
                    yield req_test
                    test_queue_end = self.env.now
                    p.wait_test.append(test_queue_end - test_queue_start)
                    lognorm = Lognormal(mean=p.mean_test, stdev=p.stdev_test)
                    sampled_test_duration = lognorm.sample()

                    yield self.env.timeout(sampled_test_duration)

                test_doc_queue_start = self.env.now

                # patients sees the doctor again for test results and follow-ups
                with self.doc.request(priority=patient.priority) as req_doc:
                    yield req_doc
                    test_doc_queue_end = self.env.now
                    p.wait_test_doc.append(doc_queue_end - doc_queue_start)

                    lognorm = Lognormal(
                        mean=p.mean_doc_consult, stdev=p.stdev_doc_consult)
                    sampled_test_doc_duration = lognorm.sample()

                    yield self.env.timeout(sampled_test_doc_duration)

                # 20% of patients needs to be hospitalised hence wait for an impatient bed
                ip_prob = random.uniform(0, 1)
                if ip_prob < 0.2:
                    sampled_ip_duration = random.expovariate(
                        1.0 / p.mean_ip_wait)
                    yield self.env.timeout(sampled_ip_duration)

            # 90% of patients need medications
            med_prob = random.uniform(0, 1)
            if med_prob < 0.9:
                med_queue_start = self.env.now

                with self.med.request() as req_med:
                    yield req_med
                    med_queue_end = self.env.now
                    p.wait_med.append(med_queue_end - med_queue_start)
                    lognorm = Lognormal(
                        mean=p.mean_med_collect, stdev=p.stdev_med_collect)
                    sampled_med_duration = lognorm.sample()

                    yield self.env.timeout(sampled_med_duration)
        # else patient leave the system

        if (patient.triage_outcome == 'not_urgent'):
            not_urgent_attend_start = self.env.now

            with self.doc_not_urgent.request() as req_not_urgent:
                yield req_not_urgent
                not_urgent_doc_queue_end = self.env.now
                p.wait_doc_not_urgent.append(
                    not_urgent_doc_queue_end - not_urgent_attend_start)
                # sample consult time
                lognorm = Lognormal(mean=p.mean_doc_noturgent,
                                    stdev=p.stdev_doc_noturgent)
                sampled_consult_duration = lognorm.sample()
                yield self.env.timeout(sampled_consult_duration)

            # 50% of patients from not urgent needs a test
            test_prob = random.uniform(0, 1)
            if test_prob < 0.5:
                # request a test
                test_queue_start = self.env.now

                with self.test.request(priority=patient.priority) as req_test:
                    yield req_test
                    test_queue_end = self.env.now
                    p.wait_test.append(test_queue_end - test_queue_start)
                    lognorm = Lognormal(mean=p.mean_test, stdev=p.stdev_test)
                    sampled_test_duration = lognorm.sample()

                    yield self.env.timeout(sampled_test_duration)

                test_doc_queue_start = self.env.now

                with self.doc_not_urgent.request() as req_doc_not_urgent:
                    yield req_doc_not_urgent
                    test_doc_queue_end = self.env.now
                    p.wait_test_doc.append(
                        test_doc_queue_end - test_doc_queue_start)
                    # sample consult time from lognormal
                    lognorm = Lognormal(
                        mean=p.mean_doc_consult, stdev=p.stdev_doc_consult)
                    sampled_test_doc_duration = lognorm.sample()
                    yield self.env.timeout(sampled_test_doc_duration)

                # 20% needs to be hospitalised
                ip_prob = random.uniform(0, 1)
                if ip_prob < 0.2:
                    sampled_ip_duration = random.expovariate(
                        1.0 / p.mean_ip_wait)
                    yield self.env.timeout(sampled_ip_duration)

            # 90% need medications
            med_prob = random.uniform(0, 1)
            if med_prob < 0.9:
                med_queue_start = self.env.now

                with self.med.request() as req_med:
                    yield req_med
                    med_queue_end = self.env.now
                    p.wait_med.append(med_queue_end - med_queue_start)
                    lognorm = Lognormal(
                        mean=p.mean_med_collect, stdev=p.stdev_med_collect)
                    sampled_med_duration = lognorm.sample()

                    yield self.env.timeout(sampled_med_duration)

        # else leave the system

        p.total_time_in_hospital.append(self.env.now - enter_system_time)

    # method to run the entire simulation
    def run(self):
        self.env.process(self.generate_patients())
        self.env.run(p.duration)
        # return mean(p.wait_triage), mean(p.wait_test), mean(p.wait_doc), mean(p.wait_doc_not_urgent)


for run in range(p.runs):
    my_ae_model = HospitalAE()
    my_ae_model.run()
    totaltime += sum(p.total_time_in_hospital)
    totalpatients += len(p.total_time_in_hospital)
    p.wait_test = []
    p.wait_triage = []
    p.wait_doc = []
    p.wait_test_doc = []
    p.wait_med = []
    p.wait_doc_not_urgent = []
    p.total_time_in_hospital = []
    # print(p.interarrival)
print(totaltime/totalpatients)


# ssef forms
# biblio
