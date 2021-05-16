# coding=utf-8

class TrueClass(dict):

    def __init__(self, *arg, **kw):
        '''

         Dictonary containing matches true class
        '''
        super(TrueClass, self).__init__(*arg, **kw)
        self.__last_alias = {}


    def __update_alias(self, alias):
        '''
        Used to differentiate in case of duplicated alias
        :param alias: Name inserted by the programmer
        :return: The same name concatenated with a incrementing number(example: gonçalo->gonçalo1,gonçalo1->gonçalo2,etc..)
        '''
        if self.__last_alias.__contains__(alias):

            value = self.__last_alias.__getitem__(alias)
            if (value == ''):
                value=1
            else:
                value +=1
            self.__last_alias.update({alias: value})
            return alias + str(value)
        else:
            self.__last_alias.update({alias: ''})
            return alias

    def __get_last_alias(self,alias):
        '''
        Get the last incremented number from alias
        :param alias: Name inserted by the programmer
        :return: last incremented number from alias
        '''
        value = self.__last_alias.__getitem__(alias)
        return alias + str(value)

    def __append_into_dict(self, alias, identificador):
        '''
        Used to insert the true matches
        :param alias: Name inserted by the programmer
        :param identificador: Id from Tracking System
        :return: void
        '''
        if self.__last_alias.__contains__(alias):
            if len(self.__getitem__(self.__get_last_alias(alias))) == 2:
                self.__setitem__(self.__update_alias(alias), [identificador])
            else:
                value = self.__getitem__(self.__get_last_alias(alias))[0]
                self.update({self.__get_last_alias(alias): [value, identificador]})

        else:
            self.__setitem__(alias, [identificador])
            self.__update_alias(alias)

    def __alternative_trueClass(self,alias,identificador):
        self.__setitem__(identificador,alias)

    def request_alias(self,identificador,type="alternative"):
        '''
        Request the programmer to insert a alias
        :param identificador: Id from Tracking System
        :return: void
        '''
        alias = input("Insert alias ")
        if type == "alternative":self.__alternative_trueClass(alias,identificador)

        else:self.__append_into_dict(alias,identificador)





