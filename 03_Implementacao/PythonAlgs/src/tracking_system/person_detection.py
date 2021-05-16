'''
PersonDetection interface.
'''

__authors__ = "Gonçalo Ferreira, Gonçalo Adolfo, Frederico Costa"
__email__ = "a43779@alunos.isel.pt, goncaloadolfo20@gmail.com, fredcosta.uni@gmail.com"


class PersonDetection:

    def get_persons(self, frame):
        '''
        Abstract method!
        Detects persons in received frame.

        Args:
        -----
            frame (ndarray) : frame intended to detect people

        Return:
        -------
            (list) : list of persons detected
        '''
        raise NotImplementedError

    @property
    def debug_frame(self):
        '''
        Abstract method!
        Obtains a debug frame with persons bounding boxes.

        Return:
        -------
            (ndarray) : frame
        '''
        return NotImplementedError

