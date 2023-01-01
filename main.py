import simpy
import random
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# import lognorm as Lognorm but keep having bugs so just random.expovariate

# class to hold global parameters - used to alter model dynamics


class p:
    # interarrival mean for exponential distribution sampling
    inter = 5
    # mean and stdev for lognormal function which converts to Mu and Sigma used to sample from lognoral distribution
    mean_doc_consult = 30
    stdev_doc_consult = 10
    mean_nurse_triage = 10
    stdev_nurse_triage = 5

    number_docs = 3
    number_nurses = 2
    number_tests = 7
    number_med = 3

    # mean time to wait for test and an inpatient bed if decide to admit
    mean_test_wait = 60
    mean_ip_wait = 20
    mean_med_collect = 5

    # simulation run metrics
    warm_up = 120
    sim_duration = 480
    number_of_runs = 1

    # some placeholders used to track wait times for resources
    wait_triage = []
    wait_test = []
    wait_doc = []
    wait_test_doc = []
    wait_med = []

    # not_urgent metrics
    mean_doc_consult_not_urgent = 20
    stdev_doc_consult_not_urgent = 7

    number_docs_not_urgent = 2
    number_nurses_not_urgent = 3
    not_urgent_number_tests = 5

    wait_doc_not_urgent = []


# class representing patients coming in
class AEPatient:
    def __init__(self, p_id) -> None:
        self.p_id = p_id
        self.time_in_system = 0

    def set_come(self):
        self.come = random.choices(['ambulance', 'walkin'], [0.12, 0.88])[0]

    def set_priority(self):
        # set priority according to weighted random choices - most are moderate in priority

        # if come by ambulance, 98.5% is urgent
        if self.come == 'ambulance':
            self.priority = random.choices([1, 2, 3, 4, 5], [0.1, 0.2, 0.4, 0.285, 0.015])[
                0]  # change if necessary
        else:
            self.priority = random.choices(
                [1, 2, 3, 4, 5], [0.1, 0.2, 0.4, 0.2, 0.1])[0]

    def set_triage_outcome(self):
        # decision tree - if priority 5, go to not urgent or home. Higher priority go to AE
        if self.priority < 5:
            self.triage_outcome = 'AE'
        elif self.priority == 5:  # of those who are priority 5, 20% will go home with advice, 80% go to 'not_urgent'
            self.triage_outcome = random.choices(
                ['home', 'not_urgent'], [0.2, 0.8])[0]


# class representing AE model
class AEModel:
    def __init__(self) -> None:
        # set up simpy env
        self.env = simpy.Environment()
        self.patient_counter = 0
        # set docs and test as priority resources - urgent patients get seen first
        self.doc = simpy.PriorityResource(self.env, capacity=p.number_docs)
        self.nurse = simpy.Resource(self.env, capacity=p.number_nurses)
        self.test = simpy.PriorityResource(self.env, capacity=p.number_tests)
        self.med = simpy.Resource(self.env, capacity=p.number_med)
        # not_urgent resources - all FIFO
        self.doc_not_urgent = simpy.Resource(
            self.env, capacity=p.number_docs_not_urgent)
        self.nurse_not_urgent = simpy.Resource(
            self.env, capacity=p.number_nurses_not_urgent)
        self.test_not_urgent = simpy.Resource(
            self.env, capacity=p.not_urgent_number_tests)

    # a method that generates AE arrivals - remember this sits in the AEModel class

    def generate_ae_arrivals(self):
        while True:
            # add patient
            self.patient_counter += 1

            # create class of AE patient and give ID
            ae_p = AEPatient(self.patient_counter)

            # simpy runs the attend ED method
            # this method is defined later
            self.env.process(self.attend_ae(ae_p))

            # Randomly sample the time to the next patient arriving to ae.
            # The mean is stored in the p class.
            sampled_interarrival = random.expovariate(1.0 / p.inter)

            # Freeze this function until that time has elapsed
            yield self.env.timeout(sampled_interarrival)

    # a method that generates AE arrivals

    def attend_ae(self, patient):
        # this is where we define the pathway through AE
        # code includes points at which we note simulation time, such as line below
        # use method below to decide triage outcome, before proceeding to AE, not_urgent or home
        patient.set_come()
        patient.set_priority()
        patient.set_triage_outcome()
        enter_system_time = self.env.now
        # if ambulance + AE no triage
        if (patient.come == 'ambulance' and patient.triage_outcome == 'AE'):
            p.wait_triage.append(0)
        else:
            enter_system_time = self.env.now
            # request a triage nurse
            # in SimPy requesting a resource can be done as below
            # the 'with' block holds the resource until all the code inside is executed
            with self.nurse.request() as req:
                # freeze until request can be met
                yield req
                triage_queue_end = self.env.now

                if self.env.now > p.warm_up:  # if past warm up period, append wait time
                    p.wait_triage.append(triage_queue_end - enter_system_time)

                # sample triage time from lognormal

                # lognorm_1 = stats.lognorm(p.mean_nurse_triage, p.stdev_nurse_triage)
                # sampled_triage_duration = lognorm_1.random.sample()
                # thiskeeps giving errors so i just dont care, the sample duration can fix later
                sampled_triage_duration = random.expovariate(
                    1.0 / p.mean_nurse_triage)

                # freeze for sampled time - simulating a triage period with the nurse
                yield self.env.timeout(sampled_triage_duration)

        if patient.triage_outcome == 'AE':

            # request doctor, a priority resource
            doc_queue_start = self.env.now

            # request doc with 'with' block
            with self.doc.request(priority=patient.priority) as req_doc:
                yield req_doc
                doc_queue_end = self.env.now
                p.wait_doc.append(doc_queue_end - doc_queue_start)
                # sample consult time from lognormal
                # lognorm_1 = stats.lognorm(
                #     p.mean_doc_consult, p.stdev_doc_consult)
                # sampled_consult_duration = lognorm_1.sample()
                sampled_consult_duration = random.expovariate(
                    1.0 / p.mean_doc_consult)

                yield self.env.timeout(sampled_consult_duration)
            # below of request for test
            test_prob = random.uniform(0, 1)
            if test_prob < 0.8:
                # request a test
                test_queue_start = self.env.now

                with self.test.request(priority=patient.priority) as req_test:
                    yield req_test
                    test_queue_end = self.env.now
                    p.wait_test.append(test_queue_end - test_queue_start)
                    sampled_test_duration = random.expovariate(
                        1.0 / p.mean_test_wait)
                    yield self.env.timeout(sampled_test_duration)

                test_doc_queue_start = self.env.now

                with self.doc.request(priority=patient.priority) as req_doc:
                    yield req_doc
                    test_doc_queue_end = self.env.now
                    p.wait_test_doc.append(doc_queue_end - doc_queue_start)
                    # sample consult time from lognormal
                    # lognorm_1 = stats.lognorm(
                    #     p.mean_doc_consult, p.stdev_doc_consult)
                    # sampled_consult_duration = lognorm_1.sample()
                    sampled_consult_duration = random.expovariate(
                        1.0 / p.mean_doc_consult)

                    yield self.env.timeout(sampled_consult_duration)

                # do they need 住院
                ip_prob = random.uniform(0, 1)
                if ip_prob < 0.2:
                    sampled_ip_duration = random.expovariate(
                        1.0 / p.mean_ip_wait)
                    yield self.env.timeout(sampled_ip_duration)

            # if they need medications
            med_prob = random.uniform(0, 1)
            if med_prob < 0.9:
                med_queue_start = self.env.now

                with self.med.request() as req_med:
                    yield req_med
                    med_queue_end = self.env.now
                    p.wait_med.append(med_queue_end - med_queue_start)
                    sampled_test_duration = random.expovariate(
                        1.0 / p.mean_med_collect)
                    yield self.env.timeout(sampled_test_duration)

        # else leave the system

        # if decision was not urgent, then:
        elif patient.triage_outcome == 'not_urgent':
            not_urgent_attend_start = self.env.now

            with self.doc_not_urgent.request() as req_not_urgent:
                yield req_not_urgent

                not_urgent_doc_queue_end = self.env.now
                p.wait_doc_not_urgent.append(
                    not_urgent_doc_queue_end - not_urgent_attend_start)
                # sample consult time
                # lognorm_1 = stats.lognorm(
                #     p.mean_doc_consult_not_urgent, p.stdev_doc_consult_not_urgent)
                # sampled_consult_duration = lognorm_1.sample()
                sampled_consult_duration = random.expovariate(
                    1.0 / p.number_docs_not_urgent)
                yield self.env.timeout(sampled_consult_duration)

            # same as AE per diagram
            test_prob = random.uniform(0, 1)
            if test_prob < 0.5:
                # request a test
                test_queue_start = self.env.now

                with self.test.request(priority=patient.priority) as req_test:
                    yield req_test
                    test_queue_end = self.env.now
                    p.wait_test.append(test_queue_end - test_queue_start)
                    sampled_test_duration = random.expovariate(
                        1.0 / p.mean_test_wait)
                    yield self.env.timeout(sampled_test_duration)

                test_doc_queue_start = self.env.now

                with self.doc.request(priority=patient.priority) as req_doc:
                    yield req_doc
                    test_doc_queue_end = self.env.now
                    p.wait_test_doc.append(doc_queue_end - doc_queue_start)
                    # sample consult time from lognormal
                    # lognorm_1 = stats.lognorm(
                    #     p.mean_doc_consult, p.stdev_doc_consult)
                    # sampled_consult_duration = lognorm_1.sample()
                    sampled_consult_duration = random.expovariate(
                        1.0 / p.mean_doc_consult)

                    yield self.env.timeout(sampled_consult_duration)

                # do they need 住院
                ip_prob = random.unifrom(0, 1)
                if ip_prob < 0.2:
                    sampled_ip_duration = random.expovariate(
                        1.0 / p.mean_ip_wait)
                    yield self.env.timeout(sampled_ip_duration)

            # if they need medications
            med_prob = random.uniform(0, 1)
            if med_prob < 0.9:
                med_queue_start = self.env.now

                with self.med.request() as req_med:
                    yield req_med
                    med_queue_end = self.env.now
                    p.wait_med.append(med_queue_end - med_queue_start)
                    sampled_test_duration = random.expovariate(
                        1.0 / p.mean_med_collect)
                    yield self.env.timeout(sampled_test_duration)

        # else leave the system
        # record time in system

        patient.time_in_system = self.env.now - enter_system_time

    # method to run sim
    def run(self):
        # start the first process which starts generating AE patient arrivals
        self.env.process(self.generate_ae_arrivals())
        # run the sim for the specified warm up period and duration - after which the while loop terminates
        self.env.run(until=p.warm_up + p.sim_duration)
        # return some results
        # return mean(p.wait_triage), mean(p.wait_test), mean(p.wait_doc), mean(p.wait_doc_not_urgent)


for run in range(p.number_of_runs):
    print(f"Run {run} of {p.number_of_runs}")
    my_ae_model = AEModel()
    # you would append these results to another list - see full code.
    # triage_mean, test_mean, doc_mean, not_urgent_mean =
    my_ae_model.run()
    print(p.wait_test)
    print(my_ae_model.patient_counter)
