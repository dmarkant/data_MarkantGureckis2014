import os, sys, stat
from string import *
from catlib import *
import pygame
from scipy import pi
from pygame.locals import *
from time import sleep
from lib.pypsyexp import *
from numpy import *
from numpy.linalg import *
from random import random, randint, shuffle, sample
from socket import gethostname

EXPERIMENT_NAME = 'Adaptive Category Learning'
FULLSCREENRES = (1380,900)
LAPTOPRES = (1280, 800)
host = gethostname().split('.')[0].lower()
if host in ['crush','bash','fracture','slam','shatter']:
    LAPTOP = False
    SCREENRES = FULLSCREENRES
else:
    LAPTOP = True
    SCREENRES = LAPTOPRES

CATUNKNOWN = -1
ACTIVE = 0
PASSIVE_YOKED = 1

RELATIVE_FEEDBACK = False   # if feedback should be relative membership in either category

# STIMULI
CIRCLE = 0

# Type of stimuli used. Right now only circles work, but there is some 
# legacy code below for gabors and rectangles
STIMULUS_TYPE = CIRCLE  

BOUNCE = False
TOGGLE = True
TOGGLEDIR = 0   # 0 -> horizontal, 1 -> vertical

SCALEA = 0.28       # scale factor for rendering stimulus (first dimension)
SCALEB = 0.9        # scale factor for rendering stimulus (second dimension)
TOLERANCE = 0.01    # acceptable error during passive and yoked conditions
MIN_R = 10          # minimum radius of rendered stimuli
NOISEPROB = 0      

LABEL_IMGS = ['ch1.png','ch2.png']

# CATEGORIES:=[[mean_A, cov_A],[mean_B, cov_B]]
UNI_ORIENT = [[[300,220],[[9000,0],[0,2000]]],[[300,380],[[9000,0],[0,2000]]]]
UNI_SIZE = [[[220,300],[[2000,0],[0,9000]]],[[380,300],[[2000,0],[0,9000]]]]
DIAG_NEG = [[[250,250],[[4538, -4463],[-4463, 4538]]],[[350, 350],[[4538, -4463],[-4463, 4538]]]]
DIAG_POS = [[[250,350],[[4538, 4463],[4463, 4538]]],[[350, 250],[[4538, 4463],[4463, 4538]]]]

# TRIAL TYPE
TRIAL_INSTRUCTION = 0
TRIAL_TRAIN = 1
TRIAL_TEST = 2

# TRIALS
N_BLOCKS = 8                    # number of blocks of each type
N_TRAIN_TRIALS_PER_BLOCK = 16
N_TEST_TRIALS_PER_BLOCK = 32

# COLORS
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
grey = (220,220,220)
greyer = (190, 190, 190)
blue = (0, 0, 238)
light_grey = ( 240, 240, 240)
dark_grey = (50, 50, 50)



#
# Main class, extends Experiment class from pypsyexp
#
class ActiveLearning(Experiment):
    
    def __init__(self, testcondition=None):
        Experiment.__init__(self, LAPTOP, SCREENRES, EXPERIMENT_NAME)
        pygame.mouse.set_visible(False)
        
        # setup conditions and datafile
        if testcondition is None:
            [self.cond, self.ncond, self.subj] = self.get_cond_and_subj_number('patterncode.txt')
            self.datafile = open("data/%s.dat" % self.subj, 'w')
            self.cond = self.subj % 8
        else:
            print "TESTING"
            self.cond = testcondition
            self.subj = 1000
            self.datafile = open("testoutput.dat","w")
        
        # setup counterbalancing based on the current condition
        params = [int(s) for s in open('conditions.txt','r').readlines()[self.cond].rstrip('\n').split(' ')]
        self.training_cond, self.task_cond, self.cat_type = params

        # if yoked participant, figure out who they will get data from
        if self.training_cond is PASSIVE_YOKED:
            self.yokedtarg = self.subj - 4
            self.targets = self.load_subject_samples( self.yokedtarg )

        # set up category structure based on condition
        if self.task_cond==0:       self.dists = [UNI_SIZE,UNI_ORIENT][self.cat_type]
        elif self.task_cond==1:     self.dists = [DIAG_POS,DIAG_NEG][self.cat_type]
        self.CAT = CategoryStructure('binary normal','binary normal',self.dists)

        # randomize other experiment settings
        self.CATA, self.CATB = choice([[0,1],[1,0]]) # category labels
        self.DIM1KEY, self.DIM2KEY = choice([[K_z,K_x],[K_x,K_z]]) # sampling control keys
        self.AKEY, self.BKEY = choice([[K_RMETA,K_RALT],[K_RALT,K_RMETA]]) # response keys
        self.DIM1DIR, self.DIM2DIR = choice([0,1]), choice([0,1])
        self.TEST_ORDER = sample(range(N_BLOCKS), N_BLOCKS)
        self.MIN_R, self.MIN_TH = randint(10,100), randint(30,90)

        # load test items, randomize order of presentation
        fp = open("testset_pseudo.dat","r")
        testseq = map(lambda s: map(lambda i: int(i), s.rstrip("\n").split("\t")), fp.readlines())
        randomized=[]
        for index in self.TEST_ORDER:
            subset = reshape(testseq[index],(32,2))
            order=range(len(subset))
            shuffle(order)
            newsubset=[]
            for i in order:
                newsubset.append(list(subset[i]))
            randomized.append(newsubset)            
        self.testset = list(reshape(flatten(randomized),(len(flatten(randomized))/2,2)))

        # output settings
        self.outputprefix = [self.subj, self.cond, self.training_cond, self.task_cond, self.cat_type]        
        
        s = "Subject %s in condition %s" % (self.subj, self.cond)
        if self.training_cond is PASSIVE_YOKED:
            s += ", using samples from subject %s\n" % self.yokedtarg
        else:
            s += "\n"
        s += "Training Condition: %s\n" % self.training_cond
        s += "Task: %s\n" % self.task_cond
        s += "Rule: %s\n" % self.dists
        s += "Noise: %s\n" % NOISEPROB
        s += "CATA=%s\n" % self.CATA
        s += "CATB=%s\n" % self.CATB
        s += "Sample keys: %s %s\n" % (self.DIM1KEY, self.DIM2KEY)
        s += "Resp keys: %s %s\n" % (self.AKEY, self.BKEY)
        s += "Dim direction: %s %s\n" % (self.DIM1DIR, self.DIM2DIR)
        s += "Dim minimum: %s %s\n" % (self.MIN_R, self.MIN_TH)
        s += "Test order: %s\n" % str(self.TEST_ORDER)
        
        print s
        self.datafile.writelines(s)

        # setup the display
        self.load_all_images('images')
        self.size = self.screen.get_rect() 
        self.size.w = self.size[2]
        self.size.h = self.size[3]
        self.surface = pygame.Surface( [self.size.w, self.size.h] )
        self.surface.fill(white)
        self.set_draw_area( [540,540] )        
        self.dimensions = self.dim = list(SCREENRES)
        self.start = [.25*self.dimensions[0], .25*self.dimensions[1] ] #top left border
        self.end = [.75*self.dimensions[0], .75*self.dimensions[1] ] #bottom right border
        self.li = arange( 0, 600 ).tolist()
        self.li_h = arange( 0, 600 ).tolist()            
        

    def do_exp(self, cond=None):
                    
        self.do_instructions()
        
        self.accuracy = []
        
        # alternate learning and test blocks
        for block in range(N_BLOCKS):
            if self.training_cond is ACTIVE:
                self.do_teach_block(N_TRAIN_TRIALS_PER_BLOCK)
            else:
                self.do_teach_block_passive(N_TRAIN_TRIALS_PER_BLOCK)
                
            self.do_test_block(N_TEST_TRIALS_PER_BLOCK)
        
        # thank you screen
        self.show_instructions("thankyou.png",message=False)
        self.cur_exit()
    
    
    def do_instructions(self):
        self.show_instructions("instructions-1.png",key='space',message=False)
        self.show_instructions("instructions-2.png",key='space',message=False)
        
        self.set_draw_area([540,540])        
        if self.training_cond is ACTIVE:
            
            self.show_instructions("instructions-3-active.png",key='space',message=False)
            
            if self.DIM1KEY==K_z:
                self.show_instructions("instructions-3b-active_a.png",key='space',message=False)
            else:
                self.show_instructions("instructions-3b-active_b.png",key='space',message=False)
                            
            for i in range(3):
                if self.DIM1KEY==K_z:
                    self.set_background("backdrop-active_a.png")
                else:
                    self.set_background("backdrop-active_b.png")
                    
                self.active_sample( None, True )
        
        elif self.training_cond is PASSIVE_YOKED:
            self.show_instructions("instructions-3-passive.png",key='space',message=False)
            if self.AKEY==K_RMETA:
                self.show_instructions("instructions-3b-passive_a.png",key='space',message=False)
            else:
                self.show_instructions("instructions-3b-passive_b.png",key='space',message=False)
            self.do_teach_block_passive(3, train=True)
                    
        self.clear_surface()
        
        if self.AKEY==K_RMETA:
            self.show_instructions("instructions-4_a.png",key='space',message=False)
        else:
            self.show_instructions("instructions-4_b.png",key='space',message=False)
        
        self.show_instructions("instructions-4conf.png",key='space',message=False)
        
        self.do_test_block(5, practice=True)

        self.show_instructions("instructions-6.png",key='space',message=False)
        self.show_instructions("instructions-7.png",key='space',message=False)
        self.show_instructions("instructions-5.png",key='p',message=False)
         
        # reset the draw area and display
        #self.set_draw_area( self.dimensions, [0,0] )        
        self.screen.blit(self.surface, [0,0])
        pygame.display.flip()
        
    
    def do_teach_block(self, ntrials):
        """ Training block for active conditions """
        if self.DIM1KEY==K_z:
            self.show_instructions("instructions-train-active_a.png", key='space', message=False)
        elif self.DIM1KEY==K_x:
            self.show_instructions("instructions-train-active_b.png", key='space', message=False)

        for i in range(ntrials):        

            # set different backdrop depending on key assignments
            if self.DIM1KEY==K_z:
                self.set_background("backdrop-active_a.png")
            else:
                self.set_background("backdrop-active_b.png")
                
            self.do_teach_trial()
               

    def do_teach_block_passive(self, ntrials, train=False):
        """ Training block for passive conditions """
        if not train:
            
            if self.AKEY==K_RMETA:
                self.show_instructions("instructions-train-passive_a.png",key='space',message=False)
            else:
                self.show_instructions("instructions-train-passive_b.png",key='space',message=False)
                                    
        self.clear_surface()
        
        for i in range(ntrials):
            
            self.surface = pygame.Surface( self.dimensions )
            self.surface.fill(white)
            if self.AKEY==K_RMETA:
                background=self.show_image_add(self.surface,'buttons_a.png', 0, 340)
            else:
                background=self.show_image_add(self.surface,'buttons_b.png', 0, 340)
            self.surface.blit(background, [0,0])
            
            if train:
                # randomly get target for trials during the instructions
                target = self.gen_exemplar()[0]
            else:
                # otherwise get the next target from the master list
                target = self.targets.pop(0)
            
            self.do_teach_trial_passive( target, train )
                
        self.clear_surface()
                
    
    def do_teach_trial(self, target=None):
        """ Single training trial for active conditions """
        self.lis = []
        [act_stim,target_stim,feedback,rt] = self.active_sample( target=target )
        
        print "\ntrain:", act_stim
        print "label:", feedback
        
        self.output_trial(flatten([self.outputprefix, TRIAL_TRAIN, act_stim, target_stim, feedback, rt, self.lis]),quiet=True)        

    def do_teach_trial_passive(self, sample, train=False):
        """ Single training trial for passive conditions """
        self.lis = []
                
        # present the current sample
        feedback, rt = self.passive_sample( sample, train )
        
        print "\ntrain:", sample
        print "label:", feedback
        
        if not train:
            self.output_trial(flatten([self.outputprefix, TRIAL_TRAIN, sample[0], sample[1], feedback, rt]), quiet=True)
        
        self.clear_surface()
        sleep(0.5)

    
    def do_test_block(self, ntrials, practice=False):        
        if not practice:
            if self.AKEY==K_RMETA:
                self.show_instructions("instructions-test_a.png",key='space',message=False)
            else:
                self.show_instructions("instructions-test_b.png",key='space',message=False)
                                        
        self.clear_surface()
        totalcorrect = 0     
        self.lis = []
        
        for i in range(ntrials):
            
            self.surface = pygame.Surface( self.dimensions )
            self.surface.fill(white)
            
            if self.AKEY==K_RMETA:
                background=self.show_image_add(self.surface,'buttons_a.png', 0, 340)
            else:
                background=self.show_image_add(self.surface,'buttons_b.png', 0, 340)
                            
            self.surface.blit(background, [0,0])
            
            # fixation
            self.place_text_image( self.surface, "+", 40, 0, 0, black, white)
            self.screen.blit(self.surface, [0,0])
            self.update_display(self.surface)
            
            sleep(0.5 + random()/2.)  # 1-2s ITI
            totalcorrect += self.do_test_trial( practice=practice )
        
        self.clear_surface()
        
        if not practice:
            sleep(0.2)
            
            acc = int(floor((float(totalcorrect)/ntrials)*100))
            self.accuracy.append(acc)
            
            if len(self.accuracy)>1:
                background=self.show_image_add(self.surface,"test_feedback.png",0,0)
                self.place_text_image( background, str(self.accuracy[-2])+"%", 60, 0, -150, black, white)
            else:
                background=self.show_image_add(self.surface,"test_feedback_first.png",0,0)
            
            self.surface.blit(background,[0,0])
            
            self.place_text_image( self.surface, str(acc)+"%", 80, 0, 30, black, white)
            self.screen.blit(self.surface, [0,0])
            self.update_display(self.surface)
            sleep(3)
            self.clear_surface()
        else:
            sleep(2)
        

    def do_test_trial(self, practice=False):        
        if practice:
            s = flatten(self.gen_exemplar())
        else:        
            s = self.testset.pop(0)
        test_stim = [a, b] = s[:2]
        test_label = self.test_exemplar(a, b)        
        resp, rt, conf = self.test_response( test_stim )
        
        print "\ntest:", test_stim
        print "label:", test_label
        print "resp:", resp
        print "conf:", conf
        
        if resp==test_label:
            correct=1
        else:
            correct=0
        print rt, correct
        if not practice:
            self.output_trial(flatten([self.outputprefix, TRIAL_TEST, test_stim[0], test_stim[1], resp, rt, conf]), quiet=True)     
        
        return correct
        
    def test_response(self, test):
        """ Test trial for all conditions

        test: (f1,f2) feature values for test stimulus
        """

        starttime = pygame.time.get_ticks()
        
        (a,b) = test
        self.draw_stimulus_abs( a, b )
        
        resp = None
        exit = False
        while not exit:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.cur_exit()

                elif event.type == KEYDOWN:
                    if pygame.key.get_pressed()[K_LSHIFT] and pygame.key.get_pressed()[K_BACKQUOTE]: # exit game
                        self.cur_exit()
                    elif pygame.key.get_pressed()[self.AKEY]:
                        resp = self.CATA
                        exit = True                        
                    elif pygame.key.get_pressed()[self.BKEY]:
                        resp = self.CATB
                        exit = True

        # certainty judgment
        background=self.show_image_add(self.surface,'confidence.png',0,300)              
        self.screen.blit(background, (0,0))
        pygame.display.flip()

        conf = None
        exit = False
        while not exit:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.cur_exit()
                elif event.type == KEYDOWN:
                    keys = pygame.key.get_pressed()
                    if keys[K_LSHIFT] and keys[K_BACKQUOTE]: # exit game
                        self.cur_exit()
                    elif keys[K_1] or keys[K_2] or keys[K_3] or keys[K_4] or keys[K_5]:
                        exit = True
                        if keys[K_1]:
                            conf = 1
                        elif keys[K_2]:
                            conf = 2
                        elif keys[K_3]:
                            conf = 3
                        elif keys[K_4]:
                            conf = 4
                        elif keys[K_5]:
                            conf = 5                        
        
        return resp, pygame.time.get_ticks() - starttime, conf
    
    def passive_sample(self, target, train=False):
        starttime = pygame.time.get_ticks()
        
        (a,b,label) = target
        self.draw_stimulus_abs( a, b )
        
        sleep(0.25)
        
        # show feedback
        feedback = self.provide_label_passive( target, train )
        
        # wait for S to press correct category button
        resp = None
        exit = False
        while not exit:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.cur_exit()

                elif event.type == KEYDOWN:
                    if pygame.key.get_pressed()[K_LSHIFT] and pygame.key.get_pressed()[K_BACKQUOTE]: # exit game
                        self.cur_exit()
                    elif pygame.key.get_pressed()[self.AKEY]:
                        if train:
                            resp = self.CATA
                            exit = True
                        if feedback == self.CATA:
                            resp = self.CATA
                            exit = True
                    elif pygame.key.get_pressed()[self.BKEY]:
                        if train:
                            resp = self.CATB
                            exit = True
                        if feedback == self.CATB:
                            resp = self.CATB
                            exit = True
        
        return feedback, pygame.time.get_ticks() - starttime
        
    
    def active_sample(self, target=None, train=False):
        
        # give feedback in terms of probabilities (i.e., prob. target belongs to either A or B)
        if RELATIVE_FEEDBACK:
            background=self.show_image_add(self.surface,'blank.png', 0, -500)
            background=self.show_image_add(self.surface,'blank.png', 1100, -300)  
            background=self.show_image_add(self.surface,'levels.png',0,-330)              
            self.surface.blit(background,[0,0])
        
        delta_a = self.li[randint(0,len(self.li)-1)]
        delta_b = self.li_h[randint(0,len(self.li_h)-1)]
        self.lis = []
        self.draw_stimulus_abs( delta_a, delta_b, target)        
        
        starttime = pygame.time.get_ticks()
        
        # moves mouse to middle of the screen
        pygame.event.set_grab(True)
        pygame.mouse.set_pos(self.screen.get_rect().center)
        [start_x,start_y] = pygame.mouse.get_pos()
        pygame.mouse.get_rel()
                
        a = b = 0.0
        exit = False      
        feedback = None
        moved = False
        if TOGGLE:
            active=[False,False]
        else:
            active=[True,True]
        
        while feedback is None:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.cur_exit()

                elif event.type == KEYDOWN:
                    if pygame.key.get_pressed()[K_LSHIFT] and pygame.key.get_pressed()[K_BACKQUOTE]: # exit game
                        self.cur_exit()
                    elif pygame.key.get_pressed()[K_RETURN]:
                        if target == None:
                            feedback = self.provide_label( act_stim)
                        else:
                            feedback = self.provide_label( act_stim, [target_stim, target[1]]) # get converted stimulus and label
                        
                    elif TOGGLE and pygame.key.get_pressed()[self.DIM1KEY]:
                        active=[True,False]
                    elif TOGGLE and pygame.key.get_pressed()[self.DIM2KEY]:
                        active=[False,True]
                
                elif TOGGLE and event.type == KEYUP and sum(active)>0:
                    active = [False,False]
                
                elif event.type == MOUSEBUTTONDOWN and moved==True:                     
                                       
                    if target == None:
                        feedback = self.provide_label( act_stim , train=train)
                    else:
                        feedback = self.provide_label( act_stim, target_stim, train=train) # get converted stimulus and label
                                
                elif event.type == MOUSEMOTION and sum(active)>0: 
                    moved = True
                    [a,b] = pygame.mouse.get_rel() # how much did it change from last position?

                    if TOGGLE:
                        c = [a,b][TOGGLEDIR]
                        [delta_a,delta_b] = [delta_a,delta_b] + c*array(active)
                    else:
                        delta_a += a
                        delta_b += b
                    
                    if not BOUNCE:
                        delta_a = int(max(min(delta_a,599),0))
                        delta_b = int(max(min(delta_b,599),0))
                    
                    [act_stim, target_stim] = self.draw_stimulus_abs( delta_a, delta_b, target)
                    
        # lis has the full sequence of "objects" that the user experiences
        self.clear_surface()
        sleep(0.5)
        return [act_stim, target_stim, feedback, pygame.time.get_ticks() - starttime]
    
    
    def matched_to_target(self, sample, target):        
        samp_r, samp_th = self.cart2polar(sample[0],sample[1])
        targ_r, targ_th = self.cart2polar(target[0],target[1])
        r_tol, th_tol = self.cart2polar(int(599*TOLERANCE), int(599*TOLERANCE))
        if ( targ_r+r_tol > samp_r > targ_r-r_tol ) and ( targ_th+th_tol > samp_th > targ_th-th_tol ):
            return True
        else:
            return False
    
    
    def provide_label_passive(self, sample, train=False):
        pos = -315
        label = self.test_exemplar(sample[0], sample[1])
        
        if train:
            label_img = 'unknown.png'
            feedback = CATUNKNOWN
        else:
                            
            # sometimes show incorrect label (proportional to NOISEPROB)
            if random() < NOISEPROB:
                if label is self.CATA:
                    feedback = self.CATB
                elif label is self.CATB:
                    feedback = self.CATA
            else:
                feedback = label
            
            if feedback==self.CATA:
                label_img = "ch1.png"
            else:
                label_img = "ch2.png"
            
        background=self.show_image_add(self.surface,'blank.png', 1100, pos)                
        background=self.show_image_add(self.surface, label_img, 0, pos)                
        self.surface.blit(background, [0,0])        
        self.update_display(self.surface)
        return feedback
    
    
    def provide_label( self, sample, target=None, train=False ):
        """ Display feedback about category label """

        pos = -315
        
        label = self.test_exemplar(sample[0], sample[1])
        
        if target!=None and not self.matched_to_target(sample, target[0]):
            
            message = "Your shape isnt close enough to the target. Try again "
            self.place_text_image( self.surface, message, 24, 0, pos, black, light_grey)
            self.screen.blit(self.surface, [0,0])
            self.update_display(self.surface)
            sleep(1.5)
            
            background=self.show_image_add(self.surface,'blank.png', 0, pos)
            self.surface.blit(background, [0,0])
            self.update_display(self.surface)
            
            return None
            
        # relative feedback
        if RELATIVE_FEEDBACK:
            print "relative belief ", label
            levelA = pygame.Surface([50,75*label[0]])
            levelA.fill(dark_grey)
            levelB = pygame.Surface([50,75*label[1]])            
            levelB.fill(dark_grey)
            self.surface.blit(levelA,[555,90-75*label[0]])
            self.surface.blit(levelB,[676,90-75*label[1]])
        
        # single category feedback
        else:

            if train:
                label_img = 'unknown.png'
                feedback = CATUNKNOWN
            else:
                
                # some proportion of time, show incorrect label
                if random() < NOISEPROB:
                    if label is self.CATA:
                        feedback = self.CATB
                    elif label is self.CATB:
                        feedback = self.CATA
                else:
                    feedback = label
            
                if feedback==self.CATA:
                    label_img = "ch1.png"
                else:
                    label_img = "ch2.png"
                                    
            background=self.show_image_add(self.surface,'blank.png', 1100, pos)                
            background=self.show_image_add(self.surface, label_img, 0, pos)                
            self.surface.blit(background, [0,0])
        
        self.update_display(self.surface)
        sleep(1.5)
        return feedback
    
    
    def draw_stimulus_abs(self, delta_a, delta_b, target=None, update=True):
        # if STIMULUS_TYPE == SQUARE:
        #     [act_stim, target_stim]=self.draw_square_abs( delta_x, delta_y, target)
        #     self.lis.append(act_stim)
            
        if STIMULUS_TYPE == CIRCLE:
            [act_stim, target_stim]=self.draw_circle_abs( delta_a, delta_b, target, update)
            self.lis.append(act_stim)
            
        # elif STIMULUS_TYPE == GABOR:
        #     if not self.gabor_setup:
        #         print "ERROR, gabor not set up"
        #         self.cur_exit()
        #     [act_stim, target_stim]=self.draw_gabor_abs( delta_x, delta_y, target)
        #     self.lis.append(act_stim)
            
        return act_stim, target_stim

    def cart2polar(self, a, b):        
        r = int( self.MIN_R + (599)*( SCALEA*self.li[a]/599.) )        
        th = (self.MIN_TH*pi)/180. + (self.li_h[b]/599.) * (150.*pi/180.)    
        return [r, th]
                
    ##################################################################################
    # draw_circle()
    # Draws the circle 
    # delta_x/delta_y - represent the current value of the x/y coords
    # defaults - if not none, will draw a 2nd static circle in darker grey
    ##################################################################################
    def draw_circle_abs( self, delta_x, delta_y, target=None, background=None, offset=[0,0], update=True):

        X = delta_x
        Y = delta_y
        
        if self.DIM1DIR==1:
            delta_x = 599 - delta_x
        if self.DIM2DIR==1:
            delta_y = 599 - delta_y
                    
        if target!=None:
            TX, TY = target
        
        # NOTE: all coordinates passed here are absolute (i.e., samples in the range 0-600)
        #if BOUNCE:
        #    delta_x = int(.4*self.li[int(abs(delta_x % 600))]) # radius  (multiplied by 1/2 so a 600 pixel stimulus has a radius of 300)
        #    delta_y = self.li_h[int(abs(delta_y % 600))] * (pi/800.) # ashby, queller conversion of orientation for lines seems to work better than above... no bounce back, but more 'natural'
        #else:
        #    delta_x = int(0.4*self.li[delta_x])
        #    delta_y = self.li_h[delta_y] * (pi/599.)
        delta_x, delta_y = self.cart2polar(delta_x, delta_y)
                        
        surf = pygame.Surface( self.draw_area.get_rect()[2:] )
        surf.fill(white)        
        pos = surf.get_rect().center
        
        end_coords = ( (cos(delta_y)*delta_x)+surf.get_rect().centerx, (sin(delta_y)*delta_x)+surf.get_rect().centery ) #coords for line
        end_coords2 = ( -(cos(delta_y)*delta_x)+surf.get_rect().centerx, -(sin(delta_y)*delta_x)+surf.get_rect().centery )
                
        # draws the target in the passive condition
        circle_rect = pygame.draw.circle(surf, red, pos, delta_x, 3) # the circle
        pygame.draw.line(surf, red, circle_rect.center, end_coords, 3 ) # radius line 1
        pygame.draw.line(surf, red, circle_rect.center, end_coords2, 3 ) # radius line 2        
        self.draw_area.blit(surf, [0,0])
        
        #self.place_text_image( self.surface, message, 24, 0, -(.45*LAPTOPRES[1]), black, white)
        self.surface.blit(self.draw_area, self.draw_area_rect)
        self.screen.blit(self.surface, [0,0])
        if update:
            self.update_display(self.surface)
                
        # returns actual stimulus shown in radius (pixels) and angle (radians)
        if target != None:
            return [[X, Y], [TX, TY]] # [radius, radians]
        else:
            return [[X, Y], [-1, -1]]
    
    ##################################################################################
    # draw_rectangle()
    # Draws the rectangle
    # delta_x/delta_y - represent the current value of the x/y coords
    # defaults - if not none, will draw a 2nd static rectangle in grey/blue
    #
    ##################################################################################
    def draw_rectangle_abs(self, delta_x, delta_y, defaults=None):
        print "draw_rectangle_abstraction: WARNING BUGGY AND MAY NOT BE IMPLEMENTED CORRECTLY!!!"
        
        width = self.end[0]-self.start[0]
        height = self.end[1]-self.start[1]
        line_start = self.start[0]   # unnecessary, but used later on. line_start can just be replaced w/ start[0]
        
        delta_x = self.li[int(abs(delta_x % width ))]    # picking out x/y vals from list based on scaling to width/height
        delta_y = self.li_h[int(abs(delta_y % height))] 
        
        starter_rect = Rect(start[0], start[1], width, delta_y)
        surf = pygame.Surface( self.dimensions )
        surf.fill(light_grey)
        
        pygame.draw.rect(surf,red,starter_rect, 3)
        pygame.draw.rect(surf, black, Rect(0,0,self.dimensions[0],self.dimensions[1]), 3)   # border of the area
        pygame.draw.line(surf, red, (delta_x+line_start,self.start[1]), (delta_x+line_start, delta_y+self.start[1]), 3)
        
        message = "Draw a shape and press P or Q"
        if defaults != None: # if defaults is not none, we draw a default shape that needs to be matched
            dummy_x = defaults[0][0]
            dummy_y = defaults[0][1]
            dummy_rect = Rect(0,0, width, dummy_y)
            dummy_surf = pygame.Surface( [width, dummy_y] )
            dummy_surf_rect = dummy_surf.get_rect()
            #dummy_surf.set_alpha(100)
            dummy_surf.fill(light_grey)         # might be better than making the
            dummy_surf.set_colorkey(light_grey) # surf more opaque
            pygame.draw.rect(dummy_surf, blue, dummy_rect, 3)
            pygame.draw.line(dummy_surf, black, (dummy_x, 0), (dummy_x,dummy_y), 3)
            
            message = "Match your shape to the one shown"
            surf.blit(dummy_surf, starter_rect)
        
        
        self.draw_area.blit(surf, [0,0])
        self.place_text_image( self.surface, message, 24, 0, (.45*LAPTOPRES[1]), black, white)
        self.surface.blit(self.draw_area, self.draw_area_rect)
        self.update_display(self.surface)
        
        return [delta_x, delta_y] # [ vert line, height ]
    
    ##################################################################################
    # draw_gabor()
    # Draws the gabor patches - now mostly defunt 
    # start/end - used in width/height calculation
    # delta_x/y - represent the current value of the x/y coords
    # defaults - if not none, will draw a 2nd static gabor ( not working out well )
    ##################################################################################
    def draw_gabor_abs( self, delta_x, delta_y, defaults=None ):
        print "draw_gabor_abstraction: WARNING BUGGY AND MAY NOT BE IMPLEMENTED CORRECTLY!!!"
        
        #self.setup_gabor is called in init
        self.draw_area.fill(black)
        
        width_range = abs(self.end[0] - self.start[0])
        height_range = abs(self.end[1] - self.start[1])
        
        rotate_ratio = self.range_rotate/width_range # gets values based on dimensions of draw_area
        freq_ratio = (self.range_freq - self.lower_bound)/height_range
        
        angle = rotate_ratio * (delta_x % width_range)
        freq = self.li[int(abs(delta_y % height_range))]
        
        scale = 4.0
        gabor_surface = self.draw_gabor(freq, angle, scale)
        gabor_rect = gabor_surface.get_rect()
        gabor_rect.center = self.draw_area.get_rect().center
        self.draw_area.blit(gabor_surface, gabor_rect)
        
        message = "Draw a shape and press P or Q"
        if defaults != None:
            dummy_gabor = self.draw_gabor(defaults[1], defaults[0], scale)
            dummy_gabor_rect = dummy_gabor.get_rect()
            
            place_holder = pygame.Surface([self.dimensions[0], self.dimensions[1]])
            place_holder_rect = place_holder.get_rect()
            place_holder_rect.center = self.draw_area.get_rect().center
            place_holder.set_alpha(100)
            
            dummy_gabor_rect.center = place_holder_rect.center
            
            place_holder.blit(dummy_gabor, dummy_gabor_rect) 
            self.draw_area.blit(place_holder, place_holder_rect)
            #self.draw_area.blit(place_holder, [0,0])
            message = "Match your shape to the one shown"
            
        self.place_text_image( self.surface, message, 24, 0, (-.45*LAPTOPRES[1]), black, white)
        self.surface.blit(self.draw_area, self.draw_area_rect)
        self.screen.blit(self.surface, [0,0])
        
        pygame.display.flip()
        
        return [angle, freq]


    def set_draw_area(self, dimensions, offset=[0,0]):
        self.draw_area = pygame.Surface( dimensions )
        self.draw_area_rect = self.draw_area.get_rect()
        c = self.surface.get_rect().center
        self.draw_area_rect.center = (c[0]+offset[0], c[1]+offset[1])
        self.draw_area.fill(white)
    
    
    def clear_surface(self):
        self.surface = pygame.Surface( self.dimensions )
        self.surface.fill(white)
        self.update_display(self.surface)
    
    
    def cur_exit(self):
        exit()
        SystemExit
    

    def sample_distributions(self):
        fp = open('dist.txt','w')
        ns = 1000
        for n in range(ns):
            [sample, label] = self.gen_exemplar( uniform=False )
            fp.writelines("%i\t%i\t%i\n" % (sample[0], sample[1], label))
        fp.close()
    

    def gen_exemplar(self):
        return self.CAT.generate_exemplars(1)
    
    
    def test_exemplar(self, x, y):
        A = self.dists[self.CATA]
        B = self.dists[self.CATB]
        
        pa=bvnpdf([x,y], A[0], A[1] )
        pb=bvnpdf([x,y], B[0], B[1] )
    
        if pa==pb:
            return choice([self.CATA,self.CATB])
        elif pa>pb:
            return self.CATA
        else:
            return self.CATB

    
    def generate_test_sample(self):
        return [randint(0,600),randint(0,600)]
    

    def setup_gabors(self):
        grid_w = grid_h = 50
        windowsd = 10
        self.setup_gabor(grid_w, grid_h, windowsd)
        self.gabor_setup = True
        self.lower_bound = 20.0 # used for freq
        self.range_rotate = 180.0 # 180-360 is the same
        self.range_freq = 120.0


    def set_background(self, filename):
        self.surface = self.show_image(filename, white, 0, 0)
        self.update_display(self.surface)


    def show_instructions(self, filename, npress=1, key='n', message=True):
        background = self.show_image(filename, white, 0, 0)
        self.update_display(background)

        time_stamp = pygame.time.get_ticks()

        if message:
            t = "Press 'n' to continue"
            text = self.get_text_image(pygame.font.Font(None, 26), t, black, white)
            textpos = self.placing_text(text,0,360,background)
            background.blit(text, textpos)
            self.update_display(background)

        for i in range(npress):

            if i == (npress-1)  and npress != 1:
                exp_text = "once more"
                text = self.get_text_image(pygame.font.Font(None, 26), exp_text, black, white)
                textpos = self.placing_text(text,0,385,background)
                background.blit(text, textpos)
                self.update_display(background)

            while 1:
                res = self.get_response()
                if (res == key):
                    break

        rt = pygame.time.get_ticks() - time_stamp
        #self.output_trial([self.subj, self.cond, TRIAL_INSTRUCTION, filename, rt])


    def load_subject_samples(self, subj):
        print 'loading samples from subject %s' % subj
        samples = []
                
        try: 
            data = map(lambda s: map(lambda n: float(n), s.rstrip(' \n').split(' ')), open('data/%d.dat' % subj,'r').readlines()[12:])    
            for trial in data:
                if int(trial[5])==TRIAL_TRAIN:
                    t = [trial[6],trial[7],trial[10]]
                    samples.append(map(int,t))
            return samples
        except:
            print "missing sample data for the active subject! quitting..."
            self.cur_exit()

def flatten(seq):
  res = []
  for item in seq:
    if (isinstance(item, (tuple, list))):
      res.extend(flatten(item))
    else:
      res.append(item)
  return res


##################################################################################
# main()
##################################################################################
def main(test=None):
    experiment = ActiveLearning(testcondition=test)
    experiment.do_exp()

def generate_train_set( save=True ):
    exp = ActiveLearning(testcondition=0)
    nitems = N_BLOCKS * N_TRAIN_TRIALS_PER_BLOCK

    # filename = "trainset_uniform.dat"
    # fp = open(filename,"w")
    # for i in range(nitems):
    #     sample, label = exp.gen_exemplar( uniform=True )
    #     fp.writelines("%i %i %i\n" % (sample[0], sample[1], label))
    # fp.close()
    # os.chmod(filename,stat.S_IRUSR)
    
    filename = "trainset_normal.dat"
    fp = open(filename,"w")
    d = []
    for i in range(nitems/2):
        sample, label = exp.gen_exemplar( uniform=False , label=0)
        d.append(sample)
        sample, label = exp.gen_exemplar( uniform=False , label=1)
        d.append(sample)
    
    for i in range(len(d)):
        fp.writelines("%s %s\n" % (d[i][0], d[i][1]))
    fp.close()
    os.chmod(filename,stat.S_IRUSR)

def generate_test_set( save=True, random=False ):
    exp = ActiveLearning(testcondition=0)
    nitems = N_BLOCKS * N_TEST_TRIALS_PER_BLOCK

    filename = "testset.dat"
    fp = open(filename,"w")

    if random:
        for i in range(nitems):
            sample, label = exp.gen_exemplar( uniform=True )
            fp.writelines("%i %i\n" % (sample[0], sample[1]))
    else:
        samples = []
        for i in arange(0,599,ceil(sqrt(nitems))):
            for j in arange(0,599,ceil(sqrt(nitems))):
                samples.append([i, j, exp.test_exemplar(i, j)])
        shuffle(samples)
        for s in samples:
            fp.writelines("%s %s\n" % (s[0], s[1]))
    
    fp.close()    
    # change permissions so we don't accidentally delete the test set
    os.chmod(filename,stat.S_IRUSR)

#------------------------------------------------------------
# let's start
#------------------------------------------------------------
if __name__ == '__main__':
    """ 
    Usage:
        
    Run experiment normally, pulling conditions from patterncode.txt:
    
        python exp.py 

    Test a specific condition (e.g., condition 0):

        python exp.py 0

    """
    if len(sys.argv)==2:
        main( test=int(sys.argv[1]) )
    else:
        main()
