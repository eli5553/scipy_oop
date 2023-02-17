import matplotlib.pyplot as plt
from matplotlib import font_manager
from statistics import mean
from tkinter import *
import time
import math
import seaborn as sns
import matplotlib
import pandas as pd  # more data analysis
import numpy as np  # a maths and plotting module
import simpy  # DES
import random
from PIL import Image, ImageTk

show_animation = True
hide_plots = False


# make small circles, oh no its wrong it dosent go to some specific places e.g. wait doc
# why the med just disappears
# https://www.google.com/search?q=animation++in+pythonfor+the+flow+of+patients+in+a+hospital&tbm=isch&ved=2ahUKEwiWk-rRxJH9AhUx_zgGHW8sBqsQ2-cCegQIABAA&oq=animation++in+pythonfor+the+flow+of+patients+in+a+hospital&gs_lcp=CgNpbWcQA1DFC1iDImDEI2gBcAB4AIABQogBhQSSAQIxMpgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=1avpY9bUHrH-4-EP79iY2Ao&bih=649&biw=1366&rlz=1C1ONGR_enSG1002SG1002#imgrc=fT1auiamKyArAM
# transparent background


################ SET UP ANIMATION CANVAS #################


class Train:
    def __init__(self, canvas, x1, y1, x2, y2, tag):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.canvas = canvas
        # self.train = canvas.create_rectangle(
        #     self.x1, self.y1, self.x2, self.y2, fill="red", tags=tag)
        kwargs = .5
        images = []
        root = animation
        alpha = int(.3 * 255)
        fill = "green"
        fill = root.winfo_rgb(fill) + (alpha,)
        image = Image.new('RGBA', (x2-x1, y2-y1), fill)
        images.append(ImageTk.PhotoImage(image))
        canvas.create_image(x1, y1, image=images[-1], anchor='nw')
        self.train = canvas.create_rectangle(
            x1, y1, x2, y2, fill="green", tags=tag)

        self.train_number = canvas.create_text(
            ((self.x2 - self.x1)/2 + self.x1), ((self.y2 - self.y1)/2 + self.y1), text=tag)
        self.canvas.update()

    def move_train(self, deltax, deltay):
        x = random.randrange(0, 80)
        y = random.randrange(0, 60)
        self.canvas.moveto(self.train, deltax + x, deltay + y)
        self.canvas.moveto(self.train_number, deltax + x, deltay + y)
        self.canvas.update()

    def remove_train(self):
        self.canvas.delete(self.train)
        self.canvas.delete(self.train_number)
        self.canvas.update()


class Clock:
    def __init__(self, canvas, x1, y1, x2, y2, tag):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.canvas = canvas
        self.train = canvas.create_rectangle(
            self.x1, self.y1, self.x2, self.y2, fill="#fff")
        self.time = canvas.create_text(((self.x2 - self.x1)/2 + self.x1), ((
            self.y2 - self.y1)/2 + self.y1), text="Time = "+str(tag)+"s")
        self.canvas.update()

    def tick(self, tag):
        self.canvas.delete(self.time)
        self.time = canvas.create_text(((self.x2 - self.x1)/2 + self.x1), ((
            self.y2 - self.y1)/2 + self.y1), text="Time = "+str(tag)+"s")
        self.canvas.update()


def create_clock(env):
    clock = Clock(canvas, 500, 250, 700, 300, env.now)
    while True:
        yield env.timeout(1)
        clock.tick(env.now)


if show_animation == True:
    animation = Tk()
    #bitmap = BitmapImage(file="uxbridge.bmp")

    #im = PhotoImage(file="train.gif")

    canvas = Canvas(animation, width=800, height=400)
    canvas.create_image(0, 0, anchor=NW)  # , image=im
    animation.title("Uxbridge Termini Simulation")

    canvas.pack()


# platforms
if show_animation == True:
    canvas.create_rectangle(20, 20, 100, 80, fill="light blue")
    canvas.create_rectangle(100, 20, 180, 80, fill="light blue")
    canvas.create_rectangle(180, 20, 260, 80, fill="light blue")
    canvas.create_rectangle(300, 20, 380, 80, fill="light blue")
    canvas.create_rectangle(380, 20, 460, 80, fill="light blue")

    canvas.create_rectangle(100, 100, 180, 160, fill="light blue")
    canvas.create_rectangle(180, 100, 260, 160, fill="light blue")
    canvas.create_rectangle(300, 100, 380, 160, fill="light blue")
    canvas.create_rectangle(380, 100, 460, 160, fill="light blue")
    canvas.create_rectangle(300, 180, 380, 240, fill="light blue")
    canvas.create_rectangle(380, 180, 460, 240, fill="light blue")

    canvas.create_text(60, 10, text="Triage")
    canvas.create_text(140, 10, text="wait")
    canvas.create_text(220, 10, text="doc")
    canvas.create_text(340, 10, text="wait")
    canvas.create_text(420, 10, text="test")
    canvas.create_text(140, 90, text="wait")
    canvas.create_text(220, 90, text="test doc")
    canvas.create_text(340, 90, text="wait")
    canvas.create_text(420, 90, text="ip")
    canvas.create_text(340, 170, text="wait")
    canvas.create_text(420, 170, text="med")

############ END OF CANVAS #################


# import lognorm as Lognorm but keep having bugs so just random.expovariate

# class to hold global parameters - used to alter model dynamics

class Lognormal:
    """
    Encapsulates a lognormal distirbution
    """

    def __init__(self, mean, stdev, random_seed=None):
        """
        Params:
        -------
        mean = mean of the lognormal distribution
        stdev = standard dev of the lognormal distribution
        """
        self.rand = np.random.default_rng(seed=random_seed)
        mu, sigma = self.normal_moments_from_lognormal(mean, stdev**2)
        self.mu = mu
        self.sigma = sigma

    def normal_moments_from_lognormal(self, m, v):
        '''
        Returns mu and sigma of normal distribution
        underlying a lognormal with mean m and variance v
        source: https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal
        -data-with-specified-mean-and-variance.html
        Params:
        -------
        m = mean of lognormal distribution
        v = variance of lognormal distribution

        Returns:
        -------
        (float, float)
        '''
        phi = math.sqrt(v + m**2)
        mu = math.log(m**2/phi)
        sigma = math.sqrt(math.log(phi**2/m**2))
        return mu, sigma

    def sample(self):
        """
        Sample from the normal distribution
        """
        return self.rand.lognormal(self.mu, self.sigma)


totalpatients = 0
totaltime = 0


class p:
    # interarrival mean for exponential distribution sampling
    inter = 0.5
    # mean and stdev for lognormal function which converts to Mu and Sigma used to sample from lognoral distribution
    # all the stdev are pretty useless cuz i commented out them
    mean_doc_consult = 300
    stdev_doc_consult = 100
    mean_nurse_triage = 10
    stdev_nurse_triage = 5

    number_docs = 40
    number_nurses = 35
    number_tests = 40
    number_med = 35

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
    total_time_in_hospital = []

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
            self.triage_outcome = 'not_urgent'


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
            global train
            train = Train(canvas, 1, 1, 20, 20, self.patient_counter)
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
            # request a triage nurse
            # in SimPy requesting a resource can be done as below
            # the 'with' block holds the resource until all the code inside is executed
            with self.nurse.request() as req:
                # freeze until request can be met
                yield req
                triage_queue_end = self.env.now
                train.move_train(20, 20)
                if self.env.now > p.warm_up:  # if past warm up period, append wait time
                    p.wait_triage.append(triage_queue_end - enter_system_time)

                # sample triage time from lognormal

                lognorm = Lognormal(mean=p.mean_nurse_triage,
                                    stdev=p.stdev_nurse_triage)
                sampled_triage_duration = lognorm.sample()
                # freeze for sampled time - simulating a triage period with the nurse
                yield self.env.timeout(sampled_triage_duration)

        if patient.triage_outcome == 'AE':

            # request doctor, a priority resource
            train.move_train(100, 20)
            doc_queue_start = self.env.now

            # request doc with 'with' block
            with self.doc.request(priority=patient.priority) as req_doc:

                yield req_doc
                train.move_train(180, 20)
                doc_queue_end = self.env.now
                p.wait_doc.append(doc_queue_end - doc_queue_start)
                # sample consult time from lognormal
                # lognorm_1 = stats.lognorm(
                # p.mean_doc_consult, p.stdev_doc_consult)
                # sampled_consult_duration = lognorm_1.sample()
                sampled_consult_duration = random.expovariate(
                    1.0 / p.mean_doc_consult)

                yield self.env.timeout(sampled_consult_duration)
            # below of request for test
            test_prob = random.uniform(0, 1)
            if test_prob < 0.8:
                # request a test
                train.move_train(300, 20)
                test_queue_start = self.env.now

                with self.test.request(priority=patient.priority) as req_test:
                    yield req_test
                    train.move_train(380, 20)
                    test_queue_end = self.env.now
                    p.wait_test.append(test_queue_end - test_queue_start)
                    sampled_test_duration = random.expovariate(
                        1.0 / p.mean_test_wait)
                    yield self.env.timeout(sampled_test_duration)

                test_doc_queue_start = self.env.now
                train.move_train(100, 100)
                with self.doc.request(priority=patient.priority) as req_doc:
                    yield req_doc
                    test_doc_queue_end = self.env.now
                    train.move_train(180, 100)
                    p.wait_test_doc.append(doc_queue_end - doc_queue_start)
                    # sample consult time from lognormal
                    lognorm = Lognormal(
                        mean=p.mean_doc_consult, stdev=p.stdev_doc_consult)
                    sampled_triage_duration = lognorm.sample()
                    # sampled_consult_duration = lognorm_1.sample()

                    yield self.env.timeout(sampled_consult_duration)

                # do they need 住院
                ip_prob = random.uniform(0, 1)
                if ip_prob < 0.2:
                    train.move_train(300, 100)
                    sampled_ip_duration = random.expovariate(
                        1.0 / p.mean_ip_wait)
                    yield self.env.timeout(sampled_ip_duration)
                    train.move_train(380, 100)

            # if they need medications
            med_prob = random.uniform(0, 1)
            if med_prob < 0.9:
                med_queue_start = self.env.now
                train.move_train(300, 180)

                with self.med.request() as req_med:
                    yield req_med
                    train.move_train(380, 180)
                    med_queue_end = self.env.now
                    p.wait_med.append(med_queue_end - med_queue_start)
                    sampled_test_duration = random.expovariate(
                        1.0 / p.mean_med_collect)
                    yield self.env.timeout(sampled_test_duration)
        # else leave the system
            else:
                train.move_train(10, 230)

        # if decision was not urgent, then:
        if (patient.triage_outcome == 'not_urgent'):
            not_urgent_attend_start = self.env.now

            with self.doc_not_urgent.request() as req_not_urgent:
                train.move_train(100, 20)
                yield req_not_urgent
                train.move_train(180, 20)
                not_urgent_doc_queue_end = self.env.now
                p.wait_doc_not_urgent.append(
                    not_urgent_doc_queue_end - not_urgent_attend_start)
                # sample consult time
                lognorm = Lognormal(
                    mean=p.mean_doc_consult_not_urgent, stdev=p.stdev_doc_consult_not_urgent)
                sampled_consult_duration = lognorm.sample()
                yield self.env.timeout(sampled_consult_duration)

            # same as AE per diagram
            test_prob = random.uniform(0, 1)
            if test_prob < 0.5:
                # request a test
                train.move_train(300, 20)
                test_queue_start = self.env.now

                with self.test.request(priority=patient.priority) as req_test:
                    yield req_test
                    train.move_train(380, 20)
                    test_queue_end = self.env.now
                    p.wait_test.append(test_queue_end - test_queue_start)
                    sampled_test_duration = random.expovariate(
                        1.0 / p.mean_test_wait)
                    yield self.env.timeout(sampled_test_duration)

                test_doc_queue_start = self.env.now
                train.move_train(100, 100)
                with self.doc.request(priority=patient.priority) as req_doc:
                    yield req_doc
                    train.move_train(180, 100)
                    test_doc_queue_end = self.env.now
                    p.wait_test_doc.append(
                        test_doc_queue_end - test_doc_queue_start)
                    # sample consult time from lognormal
                    lognorm = Lognormal(
                        mean=p.mean_doc_consult, stdev=p.stdev_doc_consult)
                    sampled_consult_duration = lognorm.sample()
                    yield self.env.timeout(sampled_consult_duration)

                # do they need 住院
                ip_prob = random.uniform(0, 1)
                if ip_prob < 0.2:
                    train.move_train(300, 100)
                    sampled_ip_duration = random.expovariate(
                        1.0 / p.mean_ip_wait)
                    yield self.env.timeout(sampled_ip_duration)
                    train.move_train(380, 100)

            # if they need medications
            med_prob = random.uniform(0, 1)
            if med_prob < 0.9:
                med_queue_start = self.env.now
                train.move_train(300, 180)

                with self.med.request() as req_med:
                    yield req_med
                    train.move_train(380, 180)
                    med_queue_end = self.env.now
                    p.wait_med.append(med_queue_end - med_queue_start)
                    sampled_test_duration = random.expovariate(
                        1.0 / p.mean_med_collect)
                    yield self.env.timeout(sampled_test_duration)

        # else leave the system
            else:
                train.move_train(10, 230)
        # record time in system

        #patient.time_in_system = self.env.now - enter_system_time
        p.total_time_in_hospital.append(self.env.now - enter_system_time)

# wobuzhidao
    # method to run sim
    def run(self):
        # start the first process which starts generating AE patient arrivals
        self.env.process(self.generate_ae_arrivals())
        self.env.process(create_clock(self.env))
        # run the sim for the specified warm up period and duration - after which the while loop terminates
        self.env.run(until=p.warm_up + p.sim_duration)
        # return some results
        # return mean(p.wait_triage), mean(p.wait_test), mean(p.wait_doc), mean(p.wait_doc_not_urgent)


for run in range(p.number_of_runs):
    my_ae_model = AEModel()
    # you would append these results to another list - see full code.
    # triage_mean, test_mean, doc_mean, not_urgent_mean =
    my_ae_model.run()
    totaltime += sum(p.total_time_in_hospital)
    totalpatients += len(p.total_time_in_hospital)
    print(f"waiting time for doc: {p.wait_doc}")
    p.wait_test = []
    p.wait_triage = []
    p.wait_doc = []
    p.wait_test_doc = []
    p.wait_med = []
    p.wait_doc_not_urgent = []
    p.total_time_in_hospital = []

print(totaltime/totalpatients)
